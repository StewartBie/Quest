"""Paired PG-19 next-token perplexity on the Blackwell Quest backend.

Each document starts with an empty cache. Feed x[t] and score x[t+1],
teacher-forced from the first token; never dense-prefill a scored prefix.
Budgets describe KV reads, not eviction or reduced cache storage.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .attention import PagedCache, Recorder, install
from .run import forward_tokens, load, manifest, write_json


PG19_REVISION = "c5e39bf32e33f9111323aa68d7d9000d22722035"
PAGE_SIZE = 16


def metrics(total_nll, count):
    if count <= 0 or not math.isfinite(total_nll):
        raise ValueError("Need a positive token count and finite NLL")
    mean = total_nll / count
    return dict(scored_tokens=count, total_nll=total_nll, mean_nll=mean,
                perplexity=math.exp(mean) if mean < 709 else float("inf"))


def context_limit(config):
    explicit = getattr(config, "max_sequence_length", None)
    if explicit:
        return explicit
    base = config.max_position_embeddings
    scaling = getattr(config, "rope_scaling", None)
    if scaling and scaling.get("type", scaling.get("rope_type")) == "linear":
        return int(base * scaling["factor"])
    return base


@torch.inference_mode()
def score_document(model, input_ids, token_budget, emit, log_every=256):
    """emit receives one row per target; input_ids include one extra target token."""
    if (input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2
            or token_budget < 0 or token_budget % PAGE_SIZE or log_every < 1):
        raise ValueError("Expected [1, T>=2] IDs and a nonnegative, page-aligned budget")
    install(model, Recorder(), page_budget=token_budget // PAGE_SIZE)
    cache = PagedCache(input_ids.shape[1] - 1, page_size=PAGE_SIZE)
    count = input_ids.shape[1] - 1
    losses = torch.empty(count, device=input_ids.device, dtype=torch.float32)
    total, flushed = 0.0, 0
    start = time.perf_counter()
    try:
        for t in range(count):
            logits = forward_tokens(model, cache, input_ids[:, t:t+1])
            losses[t] = F.cross_entropy(logits[:, -1].float(), input_ids[:, t+1], reduction="sum")
            if (t + 1) % log_every == 0 or t + 1 == count:
                values = losses[flushed:t+1].cpu().tolist()
                for position, value in enumerate(values, start=flushed + 1):
                    if not math.isfinite(value):
                        raise ValueError(f"Non-finite NLL at target {position}")
                    total += value
                    emit(dict(target_position=position, context_tokens=position,
                              nll=value, cumulative_nll=total,
                              cumulative_ppl=math.exp(total / position)))
                flushed = t + 1
                print(f"budget={token_budget or 'full'} targets={flushed}/{count} "
                      f"ppl={math.exp(total / flushed):.6f} "
                      f"elapsed_s={time.perf_counter()-start:.1f}", flush=True)
        return dict(metrics(total, count), elapsed_seconds=time.perf_counter()-start)
    finally:
        # C ABI plans own allocations outside PyTorch; release per document.
        torch.cuda.synchronize(input_ids.device)
        model.probe_native.close()
        del cache


def summarize(documents):
    """Token-weighted corpus PPL; never average document perplexities."""
    budgets = sorted({r["token_budget"] for r in documents})
    result = []
    expected = {r["document_index"]: (r["input_sha256"], r["scored_tokens"])
                for r in documents if r["token_budget"] == 0}
    if not expected:
        raise ValueError("Full-cache baseline is required")
    for budget in budgets:
        rows = [r for r in documents if r["token_budget"] == budget]
        actual = {r["document_index"]: (r["input_sha256"], r["scored_tokens"]) for r in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError("Budgets must score identical documents and targets exactly once")
        result.append(dict(token_budget=budget, page_budget=budget // PAGE_SIZE,
            mode="full" if budget == 0 else "quest", documents=len(rows),
            **metrics(math.fsum(r["total_nll"] for r in rows), sum(r["scored_tokens"] for r in rows))))
    baseline = result[0]
    for row in result:
        row["delta_ppl"] = row["perplexity"] - baseline["perplexity"]
        row["relative_ppl_change_percent"] = (row["perplexity"] / baseline["perplexity"] - 1) * 100
        row["delta_mean_nll"] = row["mean_nll"] - baseline["mean_nll"]
    return result


def plot_curve(folder, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for doc, budget in sorted({(r["document_index"], r["token_budget"]) for r in curves}):
        rows = [r for r in curves if r["document_index"] == doc and r["token_budget"] == budget]
        ax.plot([r["context_tokens"] for r in rows], [r["cumulative_ppl"] for r in rows],
                label=f"doc {doc}: " + ("full" if budget == 0 else f"Quest {budget}"))
    ax.set(xlabel="Context tokens (also number of scored targets)",
           ylabel="Cumulative perplexity", title="PG-19: full KV vs Quest selected KV")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "perplexity.svg")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model", default="data/models/longchat-7b-v1.5-32k")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", default="emozilla/pg19-test")
    p.add_argument("--dataset-revision", default=PG19_REVISION)
    p.add_argument("--dataset-cache", default="data/pg19-cache")
    p.add_argument("--documents", type=int, default=1, help="Number of test books, each with a fresh cache")
    p.add_argument("--start-document", type=int, default=0)
    p.add_argument("--num-eval-tokens", type=int, default=30000, help="Maximum scored next-token targets per book")
    p.add_argument("--token-budgets", nargs="+", type=int, default=[0, 4096], help="0=full; others must be multiples of 16")
    p.add_argument("--log-every", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if (args.documents < 1 or args.start_document < 0 or args.num_eval_tokens < 1
            or args.log_every < 1 or 0 not in args.token_budgets
            or any(b < 0 or b % PAGE_SIZE for b in args.token_budgets)
            or len(set(args.token_budgets)) != len(args.token_budgets)):
        p.error("Require positive counts, unique page-aligned budgets including 0, and nonnegative start")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if (config.model_type != "llama" or config.num_attention_heads != config.num_key_value_heads
            or config.hidden_size // config.num_attention_heads not in (64, 128)
            or getattr(config, "pretraining_tp", 1) != 1):
        p.error("Backend supports Llama MHA, head_dim 64/128, pretraining_tp=1")
    if args.num_eval_tokens > context_limit(config):
        p.error(f"Requested context exceeds model limit {context_limit(config)}; no sliding-window reset is applied")
    # Existing loader and C ABI both use logical cuda:0. Select a physical card
    # with CUDA_VISIBLE_DEVICES, keeping the current device consistent.
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    # Existing provenance helper expects JSON-serializable arguments.
    folder = args.output
    args.output = str(folder)
    provenance = manifest(args)
    provenance.update(task="pg19_perplexity", status="running", protocol=dict(
        scoring="feed x[t], score x[t+1]; all positions teacher-forced; first target index 1",
        aggregation="exp(sum token NLL / total scored tokens)", cache_reset="between books only",
        tokenizer="add_special_tokens=True; same saved input IDs for every budget",
        page_size=PAGE_SIZE, first_dense_layers=2, tail="mandatory; included in budget",
        budget="KV read budget; complete KV is retained", num_eval_tokens="per-document maximum",
        reference="evaluation/pg19/ppl_eval.py evaluates first test book; scripts/ppl_eval.sh uses 30000 targets",
        caveat="Blackwell adapted backend; not the complete PG-19 test-set or paper reproduction unless explicitly evaluated"))
    write_json(folder / "manifest.json", provenance)
    from datasets import load_dataset
    dataset = load_dataset(args.dataset, revision=args.dataset_revision, split="test", cache_dir=args.dataset_cache)
    if args.start_document + args.documents > len(dataset):
        raise ValueError(f"Dataset contains only {len(dataset)} documents")
    provenance["dataset_fingerprint"] = dataset._fingerprint
    write_json(folder / "manifest.json", provenance)
    tokenizer, model = load(args)
    documents, curves = [], []
    try:
        for index in range(args.start_document, args.start_document + args.documents):
            item = dataset[index]
            ids = tokenizer(item["text"], add_special_tokens=True, truncation=False)["input_ids"]
            available = len(ids) - 1
            ids = ids[:args.num_eval_tokens + 1]
            if len(ids) < 2:
                raise ValueError(f"Document {index} has no next-token targets")
            digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            write_json(folder / f"input-{index}.json", dict(document_index=index,
                text_sha256=hashlib.sha256(item["text"].encode()).hexdigest(),
                input_sha256=digest, input_ids=ids, available_targets=available,
                scored_tokens=len(ids)-1, metadata={k: v for k, v in item.items() if k != "text"}))
            inputs = torch.tensor([ids], device="cuda:0", dtype=torch.long)
            for budget in args.token_budgets:
                with (folder / f"nll-doc{index}-budget{budget}.jsonl").open("w") as f:
                    def emit(row):
                        row = dict(row, document_index=index, token_budget=budget)
                        f.write(json.dumps(row) + "\n")
                        if row["target_position"] % args.log_every == 0 or row["target_position"] == len(ids)-1:
                            f.flush()
                            curves.append(row)
                    scores = score_document(model, inputs, budget, emit, args.log_every)
                row = dict(scores, document_index=index, input_sha256=digest, token_budget=budget,
                           sparse_steps=max(0, len(ids)-1-budget) if budget else 0)
                documents.append(row)
                write_json(folder / "documents.json", documents)
            del inputs
        summary = summarize(documents)
        write_json(folder / "summary.json", summary)
        write_json(folder / "curve.json", curves)
        with (folder / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        plot_curve(folder, curves)
        provenance["status"] = "complete"
        write_json(folder / "manifest.json", provenance)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if hasattr(model, "probe_native"):
            torch.cuda.synchronize()
            model.probe_native.close()


if __name__ == "__main__":
    main()
