"""Summarize timing, same-head temporal reuse and logical cross-head overlap.

Cross-head/layer logical overlap is NOT reuse of identical KV bytes in MHA.
CUDA event intervals are not pure kernel times. Profile summaries use the
actual GPU kernel events and correlate them with CPU stage annotations.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


def rows(path):
    if path.exists():
        with path.open() as f:
            for line in f:
                yield json.loads(line)


def stats(values):
    return dict(n=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95))) if values else {}


def config(row):
    return row["context_length"], row["page_budget"], row["backend"]


def profile_summary(path):
    events = json.loads(path.read_text())["traceEvents"]
    stages = sorted((e for e in events if e.get("ph") == "X" and e.get("name", "").startswith("quest/layer_")),
                    key=lambda e: e["ts"])
    starts = [e["ts"] for e in stages]
    correlation_stage = {}
    for e in events:
        if e.get("cat") not in ("cuda_runtime", "cuda_driver"):
            continue
        i = bisect.bisect_right(starts, e["ts"]) - 1
        if i >= 0 and stages[i]["ts"] <= e["ts"] <= stages[i]["ts"] + stages[i]["dur"]:
            correlation_stage[e.get("args", {}).get("correlation")] = stages[i]["name"]
    kernels = [e for e in events if e.get("cat") == "kernel" and "dur" in e]
    if not kernels:
        return dict(file=path.name, error="No GPU kernel events; profiler capture unavailable")
    groups = defaultdict(list)
    invocations = defaultdict(float)
    names = Counter()
    for e in kernels:
        call = correlation_stage.get(e.get("args", {}).get("correlation"), "unattributed")
        stage = call.rsplit("/", 1)[-1]
        groups[stage].append(e["dur"])
        if call != "unattributed":
            invocations[call] += e["dur"]
        names[e["name"]] += e["dur"]
    # GPU work union includes memcpy/memset, not only compute kernels.
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    spans = sorted((e["ts"], e["ts"] + e["dur"]) for e in gpu)
    first, end = spans[0]
    busy = 0
    start = first
    gaps = []
    for lo, hi in spans[1:]:
        if lo > end:
            busy += end - start
            gaps.append(lo - end)
            start = lo
        end = max(end, hi)
    busy += end - start
    sync = [e["dur"] for e in events if e.get("cat") == "cuda_runtime"
            and "Synchronize" in e.get("name", "") and "dur" in e]
    per_call = defaultdict(list)
    per_layer = defaultdict(list)
    for call, duration in invocations.items():
        _, layer, step, stage = call.split("/")
        per_call[stage].append(duration)
        per_layer[(layer, stage)].append(duration)
    return dict(file=path.name, kernel_count=len(kernels), gpu_span_us=end-first,
        gpu_work_union_us=busy, gpu_gap_us=(end-first)-busy,
        gpu_gap_fraction=1-busy/(end-first), gap_us=stats(gaps),
        cpu_cuda_synchronize_us=sum(sync),
        stage_kernel_us={k: dict(stats(v), total=sum(v)) for k, v in groups.items()},
        stage_invocation_kernel_us={k: stats(v) for k, v in per_call.items()},
        layer_stage_invocation_kernel_us=[dict(layer=k[0], stage=k[1], **stats(v))
                                         for k, v in per_layer.items()],
        top_kernels_total_us=names.most_common(12),
        caveat="Profile instrumentation changes scheduling; gaps are GPU idle intervals, not warp-memory stalls")


def analyze(folder):
    latency = defaultdict(list)
    for row in rows(folder / "latency.jsonl"):
        latency[(*config(row), row["mode"])].append(row["mean_step_ms"])
    timing = defaultdict(list)
    for row in rows(folder / "stages.jsonl"):
        timing[(*config(row), row["stage"])].append(row["gpu_interval_us"])

    previous = {}
    history = defaultdict(dict)
    temporal = defaultdict(lambda: defaultdict(list))
    popular = defaultdict(Counter)
    head_temporal = defaultdict(list)
    head_overlap = defaultdict(lambda: [Counter(), 0])
    for row in rows(folder / "pages.jsonl"):
        cfg = config(row)
        layer, head, step = row["layer"], row["query_head"], row["step"]
        # Retain layer/head in the address identity; exclude mandatory tail.
        identity = (*cfg, row["run"], layer, head)
        chosen = set(row["logical_pages"][:-1])
        group = (*cfg, "sparse" if row["sparse"] else "dense")
        if identity in previous:
            prev_step, old = previous[identity]
            if step == prev_step + 1:
                temporal[group]["previous_step_hit_fraction"].append(len(chosen & old)/len(chosen) if chosen else 0)
                head_temporal[(*cfg, layer, head)].append(len(chosen & old)/len(chosen) if chosen else 0)
                temporal[group]["jaccard"].append(len(chosen & old)/len(chosen | old) if chosen | old else 1)
        previous[identity] = step, chosen
        for page in chosen:
            if page in history[identity]:
                temporal[group]["reuse_gap_steps"].append(step-history[identity][page])
            history[identity][page] = step
            popular[(*cfg, layer, head)][page] += 1
        ordered = sorted(chosen)
        if len(ordered) > 1:
            temporal[group]["adjacent_logical_pairs_fraction"].append(
                sum(b == a + 1 for a, b in zip(ordered, ordered[1:]))/(len(ordered)-1))
        key = (*cfg, row["run"], layer, step)
        head_overlap[key][0].update(chosen)
        head_overlap[key][1] += len(chosen)
    cross = defaultdict(list)
    for key, (counts, total) in head_overlap.items():
        if counts:
            cross[key[:3]].append(total/len(counts))
    output = dict(
        latency_mean_step_ms=[dict(config=list(k), **stats(v)) for k, v in latency.items()],
        stage_event_interval_us=[dict(config=list(k), **stats(v)) for k, v in timing.items()],
        temporal_reuse=[dict(config=list(k), **{name: stats(v) for name, v in metrics.items()})
                        for k, metrics in temporal.items()],
        logical_head_overlap=[dict(config=list(k), mean_head_requests_per_unique_logical_page=stats(v))
                              for k, v in cross.items()],
        caveats=["Tail excluded from reuse statistics; first two full layers reported separately",
                 "MHA heads and different layers store different KV bytes even at identical logical page numbers",
                 "Logical page adjacency is descriptive, not a bandwidth efficiency measurement",
                 "Same-head recurrence is reuse opportunity, not evidence that cache lines survived between steps"],
        profiles=[profile_summary(p) for p in sorted(folder.glob("profile-*.json"))])
    (folder / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    with (folder / "page_popularity.jsonl").open("w") as f:
        for key, counts in popular.items():
            f.write(json.dumps(dict(config=list(key), most_frequent_pages=counts.most_common(10))) + "\n")
    with (folder / "head_temporal_reuse.jsonl").open("w") as f:
        for key, values in head_temporal.items():
            f.write(json.dumps(dict(config=list(key), previous_step_hit_fraction=stats(values))) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "stage_event_interval_us"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("folder", type=Path)
    analyze(parser.parse_args().folder)
