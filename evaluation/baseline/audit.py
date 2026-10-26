"""Validate experiment completeness and trace identities before analysis."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def read_rows(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def audit(folder, layers=32, heads=32):
    manifest = json.loads((folder / "manifest.json").read_text())
    args = manifest["arguments"]
    errors = []
    def check(condition, message):
        if not condition and len(errors) < 30:
            errors.append(message)
    if args["task"] == "accuracy":
        cases = list(read_rows(folder / "cases.jsonl"))
        check(len(cases) == args["cases"], "Missing cases")
        lookup = {c["case"]: c for c in cases}
        for c in cases:
            check(len(c["input_ids"]) == args["lengths"][0], "Wrong input length")
            check(hashlib.sha256(json.dumps(c["input_ids"]).encode()).hexdigest() == c["input_sha256"], "Input hash mismatch")
        seen = set()
        counts = Counter()
        for row in read_rows(folder / "accuracy.jsonl"):
            key = (row["case"], row["page_budget"])
            check(key not in seen, f"Duplicate accuracy result: {key}")
            seen.add(key)
            c = lookup[row["case"]]
            check(c["input_sha256"] == row["input_sha256"], "Unpaired accuracy input")
            check(row["correct"] == (row["answer"] == c["passkey"]), "Wrong accuracy label")
            counts[row["page_budget"]] += 1
        check(set(counts) == set(args["budgets"]), "Missing accuracy budget")
        check(all(counts[b] == args["cases"] for b in args["budgets"]), "Incomplete accuracy results")
        expected = args["cases"] * len(args["budgets"])
        summary = dict(task="accuracy", expected_results=expected, results=len(seen))
    else:
        latency = list(read_rows(folder / "latency.jsonl"))
        profile_worker = args.get("worker_mode") == "profile"
        expected_modes = Counter({"profile": 1} if profile_worker else {"none": args["repeats"]})
        if args["instrument"] and not profile_worker:
            expected_modes.update({"timing": 1, "trace": 1})
        if args["profile"]:
            expected_modes["profile"] += 1
        groups = {}
        for row in latency:
            cfg = row["context_length"], row["page_budget"], row["backend"]
            groups.setdefault(cfg, Counter())[row["mode"]] += 1
            check(len(row["step_ms"]) == args["decode_steps"], f"Incomplete decode times: {cfg}")
            check(all(math.isfinite(v) and v > 0 for v in row["step_ms"]), "Invalid latency")
        for length in args["lengths"]:
            for budget in args["budgets"]:
                for backend in args["backends"]:
                    cfg = length, budget, backend
                    check(groups.get(cfg) == expected_modes, f"Incomplete latency modes: {cfg}")
        selected_rows = 0
        seen = set()
        if args["instrument"]:
            for row in read_rows(folder / "pages.jsonl"):
                key = tuple(row[k] for k in ("context_length", "page_budget", "backend", "run", "step", "layer", "query_head"))
                check(row["context_length"] in args["lengths"] and row["page_budget"] in args["budgets"]
                      and row["backend"] in args["backends"], "Unexpected trace configuration")
                check(row["run"] == args["repeats"] + 2, "Unexpected trace run")
                check(key not in seen, f"Duplicate trace row: {key}")
                seen.add(key)
                selected_rows += 1
                pages = row["logical_pages"]
                total = (row["sequence_length"] + row["page_size"] - 1) // row["page_size"]
                budget = total if row["layer"] < 2 or row["page_budget"] == 0 else min(total, row["page_budget"])
                check(len(pages) == budget and len(set(pages)) == budget, f"Wrong page count: {key}")
                check(pages[-1] == total - 1 and min(pages) >= 0 and max(pages) < total, f"Invalid tail/index: {key}")
                check(row["physical_pages"] == pages, "Unexpected allocator mapping")
                check(row["query_head"] == row["kv_head"] and 0 <= row["query_head"] < heads, "Unexpected MHA mapping")
                check(0 <= row["layer"] < layers and 0 <= row["step"] < args["decode_steps"], "Invalid trace coordinates")
                check(row["sequence_length"] == row["context_length"] + row["step"] + 1, "Wrong decode position")
                check(row["sparse"] == (budget < total), "Wrong sparse label")
            expected = len(args["lengths"]) * len(args["budgets"]) * len(args["backends"]) * args["decode_steps"] * layers * heads
            check(selected_rows == expected, "Missing layer/step/head trace rows")
        else:
            expected = 0
        summary = dict(task="benchmark", latency_rows=len(latency), expected_trace_rows=expected,
                       trace_rows=selected_rows, configs=len(groups))
    summary.update(passed=not errors, errors=errors)
    (folder / "audit.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    audit(args.folder)
