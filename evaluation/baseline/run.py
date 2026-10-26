"""Reproducible Passkey evaluation and paired native/split profiling sweeps.

Run from repo root: python -m evaluation.baseline.run --help
All output directories must be new, to prevent mixing experimental runs.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attention import PagedCache, Recorder, install


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def manifest(args):
    def git(*cmd):
        return subprocess.check_output(["git", *cmd], text=True).strip()
    sources = {}
    paths = list(Path("evaluation/baseline").glob("*.py"))
    paths += list(Path("kernels/minimal").glob("*"))
    paths += list(Path("kernels/include").rglob("*.cuh"))
    paths += [Path("kernels/build-sm120-gcc13/libquest_probe.so")]
    for path in sorted(paths):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    model_metadata = Path(args.model) / ".cache/huggingface/download/config.json.metadata"
    revision = model_metadata.read_text().splitlines()[0] if model_metadata.exists() else None
    return dict(arguments=vars(args), process_id=os.getpid(), parent_process_id=os.getppid(), python=platform.python_version(),
        torch=torch.__version__, transformers=transformers.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(), git_commit=git("rev-parse", "HEAD"),
        git_status=git("status", "--short"), source_sha256=sources, model_revision=revision,
        implementation="original CUDA FP16 score/partitioned decode; torch.topk; HND; SDPA prefill",
        selection="budget includes mandatory tail; first 2 layers full; physical slot == logical page",
        timing="CUDA event intervals may contain host launch gaps; split gather includes writes, not DRAM-read-only time")


def make_cases(tokenizer, length, count, seed):
    rng = random.Random(seed)
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    header = [tokenizer.bos_token_id] + enc(
        "There is an important info hidden inside a lot of irrelevant text. "
        "Find it and memorize them. I will quiz you about the important information there.\n")
    question = enc("\nWhat is the pass key? The pass key is")
    garbage = enc("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n")
    result = []
    for i in range(count):
        key = rng.randint(1, 50000)
        info = enc(f"\nThe pass key is {key}. Remember it. {key} is the pass key.\n")
        available = length - len(header) - len(info) - len(question)
        if available < 32:
            raise ValueError("Context too short for Passkey")
        filler = (garbage * ((available + len(garbage) - 1) // len(garbage)))[:available]
        depth = (0.1, 0.3, 0.5, 0.7, 0.9)[i % 5]
        offset = int(available * depth)
        ids = header + filler[:offset] + info + filler[offset:] + question
        result.append(dict(case=i, seed=seed, passkey=key, depth=depth,
            answer_token_start=len(header)+offset, answer_token_end=len(header)+offset+len(info),
            input_ids=ids, input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
    return result


def load(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.float16, attn_implementation="sdpa", device_map={"": "cuda:0"}).eval()
    return tokenizer, model


def forward_tokens(model, cache, tokens):
    # Project only final hidden position to vocabulary, avoiding huge prefill logits.
    output = model.model(input_ids=tokens, past_key_values=cache, use_cache=True)
    return model.lm_head(output.last_hidden_state[:, -1:]).float()


@torch.inference_mode()
def evaluate(args, tokenizer, model, folder):
    cases = make_cases(tokenizer, args.lengths[0], args.cases, args.seed)
    with (folder / "cases.jsonl").open("w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")
    results = []
    for budget in args.budgets:
        rec = Recorder("none")
        install(model, rec, budget)
        for case in cases:
            ids = torch.tensor([case["input_ids"]], device="cuda")
            cache = PagedCache(ids.shape[1] + args.decode_steps)
            split = ids.shape[1] - args.tail_tokens
            if split <= case["answer_token_end"]:
                raise ValueError("Passkey must be in prefilled material, before the decode tail")
            logits = forward_tokens(model, cache, ids[:, :split])
            for t in range(split, ids.shape[1]):
                logits = forward_tokens(model, cache, ids[:, t:t+1])
            generated = []
            for _ in range(args.decode_steps):
                token = logits[:, -1].argmax(-1).reshape(1, 1)
                generated.append(token.item())
                if generated[-1] == tokenizer.eos_token_id:
                    break
                logits = forward_tokens(model, cache, token)
            response = tokenizer.decode(generated, skip_special_tokens=True)
            match = re.search(r"\d+", response)
            answer = int(match.group()) if match else None
            row = dict(case=case["case"], depth=case["depth"], input_sha256=case["input_sha256"],
                context_length=ids.shape[1], page_budget=budget, token_budget=budget*16,
                mode="dense_native" if budget == 0 else "quest_native",
                passkey=case["passkey"], answer=answer, response=response,
                correct=answer == case["passkey"], tail_tokens=args.tail_tokens)
            results.append(row)
            with (folder / "accuracy.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            del cache, logits
        selected = [r for r in results if r["page_budget"] == budget]
        print(f"budget={budget} accuracy={sum(r['correct'] for r in selected)}/{len(selected)}", flush=True)
    summaries = []
    def summarize(rows):
        n, correct = len(rows), sum(r["correct"] for r in rows)
        rate, z = correct / n, 1.959963984540054
        center = (rate + z*z/(2*n))/(1+z*z/n)
        half = z*math.sqrt(rate*(1-rate)/n + z*z/(4*n*n))/(1+z*z/n)
        return dict(cases=n, correct=correct, accuracy=rate,
                    wilson_95=[max(0.0, center-half), min(1.0, center+half)])
    for b in args.budgets:
        selected = [r for r in results if r["page_budget"] == b]
        summaries.append(dict(page_budget=b, **summarize(selected), by_depth=[
            dict(depth=d, **summarize([r for r in selected if r["depth"] == d]))
            for d in sorted({r["depth"] for r in selected})]))
    write_json(folder / "accuracy_summary.json", summaries)


@torch.inference_mode()
def benchmark(args, tokenizer, model, folder):
    summary = []
    for length in args.lengths:
        case = make_cases(tokenizer, length, 1, args.seed)[0]
        ids = torch.tensor([case["input_ids"]], device="cuda")
        # Fixed teacher-forced decode token sequence shared by all budgets/backends.
        replay = tokenizer.encode(" The pass key is 12345. The answer is recorded.", add_special_tokens=False)
        replay = (replay * (args.decode_steps // len(replay) + 1))[:args.decode_steps]
        write_json(folder / f"input-{length}.json", dict(case=case, decode_ids=replay,
                   workload="real-model teacher-forced synthetic Passkey; not free generation"))
        tokens = [torch.tensor([[token]], device="cuda") for token in replay]
        for backend in args.backends:
            for budget in args.budgets:
                modes = ["none"] * args.repeats if args.worker_mode == "measure" else ["profile"]
                if args.instrument and args.worker_mode == "measure":
                    modes += ["timing", "trace"]
                # One separate warmup at exactly this context, backend, and budget.
                for run, mode in enumerate(["warmup"] + modes):
                    recorder_mode = "none" if mode == "warmup" or (mode == "profile" and args.profile_level == "minimal") else mode
                    rec = Recorder(recorder_mode)
                    rec.capture_snapshot = args.snapshot and mode == "trace" and backend == "native" and budget > 0
                    rec.context = dict(context_length=length, page_budget=budget, backend=backend, run=run)
                    install(model, rec, budget, backend)
                    cache = PagedCache(length + args.decode_steps)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    begin = time.perf_counter()
                    forward_tokens(model, cache, ids)
                    torch.cuda.synchronize()
                    prefill_ms = (time.perf_counter() - begin) * 1000
                    if mode == "profile":
                        activities = [torch.profiler.ProfilerActivity.CUDA]
                        if args.profile_level == "stages":
                            activities.insert(0, torch.profiler.ProfilerActivity.CPU)
                        profiler = torch.profiler.profile(activities=activities,
                            record_shapes=args.profile_level == "stages")
                        profiler.__enter__()
                    step_events = []
                    begin = time.perf_counter()
                    for step, token in enumerate(tokens):
                        rec.step, rec.phase = step, "replay_decode"
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        forward_tokens(model, cache, token)
                        end.record()
                        step_events.append((start, end))
                    torch.cuda.synchronize()
                    wall_ms = (time.perf_counter() - begin) * 1000
                    if mode == "profile":
                        profiler.__exit__(None, None, None)
                        profiler.export_chrome_trace(str(folder / f"profile-{length}-{budget}-{backend}.json"))
                    ms = [start.elapsed_time(end) for start, end in step_events]
                    row = dict(rec.context, mode=mode, prefill_ms=prefill_ms,
                        decode_wall_ms=wall_ms, mean_step_ms=float(np.mean(ms)),
                        median_step_ms=float(np.median(ms)), p95_step_ms=float(np.percentile(ms, 95)),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(), step_ms=ms)
                    if mode != "warmup":
                        summary.append(row)
                        with (folder / "latency.jsonl").open("a") as f:
                            f.write(json.dumps(row) + "\n")
                        rec.save(folder)
                        print(json.dumps({k: v for k, v in row.items() if k != "step_ms"}), flush=True)
                    del cache
                    gc.collect()
    write_json(folder / "latency_summary.json", summary)


def benchmark_commands(args, folder):
    """Fresh interpreters isolate CUPTI state for every profiled configuration."""
    base = [sys.executable, "-m", "evaluation.baseline.run", "benchmark",
            "--model", args.model, "--seed", str(args.seed),
            "--tail-tokens", str(args.tail_tokens), "--decode-steps", str(args.decode_steps),
            "--repeats", str(args.repeats), "--profile-level", args.profile_level]
    measure = base + ["--output", str(folder / "measure"), "--worker-mode", "measure",
                      "--lengths", *map(str, args.lengths), "--budgets", *map(str, args.budgets),
                      "--backends", *args.backends]
    if args.instrument:
        measure.append("--instrument")
    if args.snapshot:
        measure.append("--snapshot")
    commands = [measure]
    for length in args.lengths:
        for backend in args.backends:
            for budget in args.budgets:
                commands.append(base + ["--output", str(folder / f"profile-{length}-{budget}-{backend}"),
                    "--worker-mode", "profile", "--lengths", str(length),
                    "--budgets", str(budget), "--backends", backend])
    return commands


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("task", choices=("accuracy", "benchmark"))
    p.add_argument("--model", default="data/models/longchat-7b-v1.5-32k")
    p.add_argument("--output", required=True)
    p.add_argument("--lengths", nargs="+", type=int, default=[10000])
    p.add_argument("--budgets", nargs="+", type=int, default=[0, 64, 128], help="Pages including mandatory tail; 0=full")
    p.add_argument("--cases", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tail-tokens", type=int, default=64)
    p.add_argument("--decode-steps", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--backends", nargs="+", choices=("native", "split"), default=["native", "split"])
    p.add_argument("--instrument", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--worker-mode", choices=("measure", "profile"), default="measure", help=argparse.SUPPRESS)
    p.add_argument("--profile-level", choices=("stages", "minimal"), default="stages")
    p.add_argument("--snapshot", action="store_true", help="Save layer 16 step 0 Q/KV/selection during trace runs")
    args = p.parse_args()
    if min(args.lengths) <= args.tail_tokens or min(args.budgets) < 0 or args.cases < 1 or args.decode_steps < 1 or args.repeats < 1:
        p.error("Invalid lengths, budgets, cases or decode steps")
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=False)
    if args.task == "benchmark" and args.profile:
        commands = benchmark_commands(args, folder)
        index = dict(status="running", isolation="fresh interpreter per profile configuration",
                     commands=commands, outputs=[c[c.index("--output") + 1] for c in commands])
        write_json(folder / "processes.json", index)
        try:
            for command in commands:
                subprocess.run(command, check=True)
        except BaseException:
            index["status"] = "failed"
            write_json(folder / "processes.json", index)
            raise
        index["status"] = "complete"
        write_json(folder / "processes.json", index)
        return
    torch.manual_seed(args.seed)
    write_json(folder / "manifest.json", manifest(args))
    tokenizer, model = load(args)
    try:
        if args.task == "accuracy":
            evaluate(args, tokenizer, model, folder)
        else:
            benchmark(args, tokenizer, model, folder)
    finally:
        if hasattr(model, "probe_native"):
            model.probe_native.close()


if __name__ == "__main__":
    main()
