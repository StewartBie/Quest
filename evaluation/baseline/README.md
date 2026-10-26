# Quest on Blackwell: measured baseline

This entry point reproduces the core algorithm on a different software/hardware
stack. It is **not** a claim of reproducing the paper's latency or speedup.

## Implementation boundary

- Batch 1, FP16, Llama MHA only, head dimensions 64/128; tested model is
  `lmsys/longchat-7b-v1.5-32k`, revision
  `389eea141540fe7d2c314eacdfe980ae1415c6c0`.
- Original CUDA `MaxPossibleSampleWithPagedKVCache` and original partitioned
  `BatchDecodeWithPagedKVCacheWrapper`, reached via a small C ABI. The original
  handler is necessary: its last partition has special tail-page semantics.
- `torch.topk` replaces RAFT selection. Cache layout is HND, with monotonically
  allocated physical page slots. Metadata/cache updates use PyTorch. Decode
  plans are cached by shape/device/stream, instead of rebuilding every step.
- Full SDPA prefill; first two layers always use full paged attention. A page
  budget **includes the mandatory tail page**. `--budgets 0` means full attention.
  The full baseline uses the same native paged decode as Quest.
- `split` is an independent diagnostic: selected KV gather followed by SDPA.
  Gather includes indices, KV reads and output writes. Its timing is **not** the
  read time inside the original fused attention kernel. SDPA itself also reads KV.
- No GQA, padding, chunked prefill, batch > 1, training or output attentions.

## Environment and build

Validated environment: RTX PRO 6000 Blackwell (`sm_120`), Python 3.10,
PyTorch 2.7.1+cu128, Transformers 4.45.2, CUDA compiler 12.8.93, host GCC 13.
The old flash-attn 2.6.3 binary had a PyTorch ABI mismatch after upgrading torch;
it was removed. SDPA does not need that package.

```bash
export QUEST_ENV=/home/bzq/.local/envs/quest
export PATH="$QUEST_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

cmake -S kernels/minimal -B kernels/build-sm120-gcc13 \
  -DCMAKE_CUDA_COMPILER="$QUEST_ENV/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build kernels/build-sm120-gcc13 -j 4
ctest --test-dir kernels/build-sm120-gcc13 --output-on-failure
python -m evaluation.baseline.test_baseline
```

Do not reuse the old build cache pointing at `/usr/bin/nvcc` (CUDA 12.0).
Explicit host compiler selection prevents Conda GCC 14 headers being mixed with
system GCC 13 in nvcc's compiler detection.

## Accuracy

```bash
python -m evaluation.baseline.run accuracy \
  --output results/baseline/passkey-new \
  --lengths 10000 --budgets 0 64 128 --cases 20 \
  --tail-tokens 64 --decode-steps 10
```

Samples have exactly 10,000 token IDs. Five depths (10/30/50/70/90%) are repeated
with different seeded passkeys. Cases and SHA256 of their IDs are saved; every
budget sees identical IDs. The last 64 input tokens are processed one by one,
then up to 10 tokens are generated greedily. The first integer is the answer.
This is an explicit token-based adaptation of the upstream character-based
prompt/tail split. A small smoke suite is not the paper's full accuracy result.
For an expanded retrieval test, increase `--cases` to 100 or more.

## PG-19 perplexity: full KV versus Quest

```bash
CUDA_VISIBLE_DEVICES=0 python -m evaluation.baseline.pg19 \
  --model data/models/longchat-7b-v1.5-32k \
  --output results/baseline/pg19-30k \
  --num-eval-tokens 30000 --documents 1 --token-budgets 0 4096
```

The default follows the repository's original `evaluation/pg19/ppl_eval.py`
and `scripts/ppl_eval.sh`: first PG-19 test book, up to 30,000 next-token
targets. The paper discusses up to 32K context and a 4096-token budget;
use `--num-eval-tokens 32000` for that length, and `--documents 100` to
request all test books. Neither the default nor a short smoke run is a
complete reproduction of the paper's PG-19 evaluation. Short books contribute
only their available targets; exact counts are recorded per book.

Every mode receives identical saved tokenizer IDs (including tokenizer-added
special tokens). Feed token `x[t]`, score `x[t+1]`, then advance: no generation,
no unscored dense prefill, no sliding-window reset. Reset KV only between books.
PPL is `exp(sum(NLL) / scored_tokens)`, aggregated by token count, not an average
of document PPLs. The last saved ID is a label, not fed into the cache. No
prediction is scored across book boundaries. Context beyond the model limit
is rejected.

`0` means full attention using the same paged CUDA backend. Other budgets are
token counts divisible by 16, including the mandatory tail page; the first two
layers remain dense. Quest retains the complete KV cache and reads selected
pages: this does not measure cache eviction or reduced KV storage. Before the
context exceeds a budget, that mode is dense too. `documents.json` records
the number of steps where sparse selection is active (in eligible layers).

The dataset is `emozilla/pg19-test`, pinned by default to commit
`c5e39bf32e33f9111323aa68d7d9000d22722035`; downloaded data uses
`data/pg19-cache`. The model is loaded locally. Output directories must be new.
Use `CUDA_VISIBLE_DEVICES` to select a card; the process uses logical CUDA 0.

Outputs: provenance in `manifest.json`; exact IDs and text/input hashes in
`input-*.json`; per-token NLL in `nll-doc*-budget*.jsonl`; per-book metrics in
`documents.json`; token-weighted PPL, delta PPL, relative change and delta mean
NLL in `summary.json`/`summary.csv`; cumulative curves in `curve.json` and
`perplexity.svg`. Only a manifest with `status=complete` denotes a finished run.
Elapsed time includes evaluation/logging and is not a latency benchmark.

```bash
python -m evaluation.baseline.test_pg19
# Short real-data validation; contexts exceed both sparse budgets:
CUDA_VISIBLE_DEVICES=0 python -m evaluation.baseline.pg19 \
  --output results/baseline/pg19-8k --num-eval-tokens 8192 \
  --token-budgets 0 1024 4096
```

## Performance and page trace

```bash
python -m evaluation.baseline.run benchmark \
  --output results/baseline/sweep-new \
  --lengths 4096 8192 16384 --budgets 0 64 128 \
  --decode-steps 16 --repeats 3 --instrument --snapshot
python -m evaluation.baseline.analyze results/baseline/sweep-new
```

These are real-model, teacher-forced synthetic Passkey workloads: the same
decode input IDs are replayed for all budgets/backends. They are not free-form
generation throughput measurements. Every shape has a separate warmup run.
Output directories must not already exist, so runs cannot silently mix.

| Artifact | Meaning |
| --- | --- |
| `manifest.json` | environment, Git revision/status, source hashes, arguments |
| `input-*.json` | exact prompt and replay input IDs |
| `latency.jsonl` | per-step CUDA intervals, synchronized decode wall time, memory |
| `stages.jsonl` | per-layer/step score, topk, KV update, fused attention or split stages |
| `pages.jsonl` | every layer/step/query head, ordered logical/physical pages, address metadata |
| `analysis.json` | timings, temporal reuse, cross-head logical overlap, profile summaries |
| `page_popularity.jsonl` | most frequently selected pages for each layer/head/config |
| `head_temporal_reuse.jsonl` | adjacent-step retention for every layer/head/config |
| `snapshot-*.pt` | layer 16, step 0 Q/KV/selection for paired replay experiments |

With `--profile`, the output root contains `processes.json`, a `measure/`
subdirectory, and one `profile-LENGTH-BUDGET-BACKEND/` subdirectory per
configuration. Each subdirectory has its own manifest and can be passed to
`analyze`/`audit`. The parent launches fresh Python interpreters; its index
is marked complete only after all workers succeed. Use `measure/` for ordinary
latency and profile subdirectories for Chrome traces.

The `none`, `timing`, `trace`, and `profile` runs are separate. Use only `none`
for baseline model latency. Stage event intervals can include host launch gaps;
they are not pure GPU kernel execution times. Snapshots perform CPU transfers
only in trace runs. Trace mode timing is never used as baseline latency.

Physical addresses are reconstructable from cache base, page stride, head stride
and V offset. The allocator uses physical slot == logical page. Across different
heads in MHA, equal page numbers mean different KV bytes; across layers this is
also true. Temporal reuse excludes the mandatory tail and separates dense layers
from sparse layers. Recurring pages are a reuse opportunity, not observed L2 hits.

## Paired access-order experiment

```bash
python -m evaluation.baseline.replay \
  results/baseline/sweep-new/snapshot-8192-64-native.pt \
  --output results/baseline/replay.json
```

This uses identical queries and selected tokens while changing physical page
placement and/or sorting selected physical page indices. Numerical equivalence
is checked before timing. CUDA graph batches amortize CPU launch overhead.
It is a **warm repeated-input** microbenchmark; payload GB/s is derived from
requested bytes, not measured DRAM traffic. It does not reproduce inter-layer
cache eviction. Address non-contiguity alone is never labeled inefficiency.

## Waiting and hardware counters

```bash
python -m evaluation.baseline.run benchmark \
  --output results/baseline/profile-new --lengths 8192 --budgets 64 \
  --decode-steps 8 --repeats 1 --backends native --profile
python -m evaluation.baseline.analyze results/baseline/profile-new
```

The Chrome trace contains CPU annotations, CUDA runtime calls and GPU kernels.
The analyzer attributes kernels through runtime correlation IDs and reports
GPU idle gaps separately from kernel execution. CPU/GPU gaps may reflect Python
dispatch or synchronization; they are not evidence of memory stalls inside a kernel.

For lower-overhead collection, add `--profile-level minimal`: CUDA-only profiling
without stage events or record_shapes. The launcher automatically runs **each profiling configuration in a fresh process**.
The measured environment retained profiling overhead on later ordinary runs in
the same process. Do not mix those later runs into the original baseline sweep.

Before analysis, validate input pairing, latency configurations and trace coverage:

```bash
python -m evaluation.baseline.audit results/baseline/sweep-new
python -m evaluation.baseline.test_analysis
```

The replay prebuilds native decode plans on the exact CUDA stream used for graph
capture. Capturing plan creation would retain H2D copies from temporary host
vectors and cause illegal access on replay. A CUDA Graph numerical regression
check is included in `evaluation.baseline.test_baseline`.

The system Nsight Compute 2022.4 does not support Blackwell. The environment has
`$QUEST_ENV/nsight-compute-2025.1.1/ncu`, which recognizes this GPU, but the current
driver sets `RmProfilingAdminOnly=1`. It returns `ERR_NVGPUCTRPERM` for hardware
counters. No driver setting is changed by these scripts.

When an administrator can run profiling, collect counters on the replay:

```bash
sudo /home/bzq/.local/envs/quest/nsight-compute-2025.1.1/ncu \
  --set full --kernel-name 'regex:BatchDecodeWithPagedKVCacheKernel' \
  --launch-count 8 --target-processes all \
  --export /tmp/quest-replay \
  /home/bzq/.local/envs/quest/bin/python -m evaluation.baseline.replay \
  results/baseline/sweep-new/snapshot-8192-64-native.pt \
  --profile-one --output /tmp/quest-replay.json
```

Record DRAM bytes and throughput, L2 hit/sector behavior, SM activity/occupancy,
and warp stall reasons (especially long scoreboard, barrier and not selected).
Profiler replay changes cache state and overhead; do not use its wall time as
baseline latency. Until counters are available, bandwidth saturation and
in-kernel memory-wait attribution remain unverified.

## Administrator counter collection script

From the repository root, inspect commands without sudo or GPU execution:

```bash
bash scripts/profile_counters.sh --dry-run
```

Collect the default 8K/64-page snapshot (score, torch.topk including int32 cast,
and four paired fused-attention layout/order variants):

```bash
sudo bash scripts/profile_counters.sh --cache-control all
```

Alternatively run without an outer sudo: the script prompts with `sudo -v` and
elevates only profiler launches. When run with outer sudo, result ownership is
returned to SUDO_UID/SUDO_GID. It does not change driver configuration or clocks.

```bash
bash scripts/profile_counters.sh --cache-control none --include-split
```

`all` lets NCU flush caches between replay passes. `none` disables this flushing,
but is not proof of in-model cache residency. Collect them into separate new
directories. `--include-split` adds gather/SDPA diagnostics, not a decomposition
of the original fused kernel. Profiling overhead must not be treated as model
baseline latency.

Specify another saved snapshot and output directory:

```bash
sudo bash scripts/profile_counters.sh \
  --snapshot results/baseline/sweep/snapshot-16384-128-native.pt \
  --cache-control all --output results/baseline/counters-16k-128-flush
```

Default output is a new timestamped `results/baseline/counters-*` directory.
Each variant has an `.ncu-rep`, raw `.csv`, capture log and numerical-check JSON.
`summary.json` selects DRAM/L2/SM/warp-stall metrics without changing units or
merging decode and merge kernels. `manifest.json` records profiler version,
snapshot/library hashes, exact commands and cache conditions. A `SUCCESS` file
is written only after every requested report was collected and exported.

Keep other GPU jobs idle during collection. Start with `--set basic` for a quick
permission/NVTX-filter check; use the default `--set full` for warp-stall analysis.
The default GPU is physical GPU 0 (`--gpu` selects another index). Paths to the
environment's Python and NCU are explicit so sudo does not select system tools.

The script sets `TMPDIR` after sudo to the fixed, non-sticky directory
`results/baseline/ncu-tmp`. This avoids a sudo/non-sudo lock conflict at
`/tmp/nsight-compute-lock` when Linux `fs.protected_regular=2` is enabled.
The existing system lock and kernel security setting are left intact. This
script's captures share the same dedicated lock directory; run no other Nsight
Compute instance with a different TMPDIR concurrently. Failed captures print
the profiler's `==ERROR==` diagnostics directly and retain the complete log.

## Resource ownership and reproducible archives

Native score, plan creation, decode and release select the input/plan CUDA device
and restore the caller's device on return or exception. `NativeKernels` supports
`with`, idempotent `close()`, reuse after close, and automatic finalization. Close
explicitly before CUDA shutdown for deterministic cleanup. Prebuild plans on the
capture stream before CUDA Graph capture as before.

For an assertions-enabled build, use the build command above with a separate
`-B kernels/build-sm120-debug -DCMAKE_BUILD_TYPE=Debug`, then build and run ctest
in that directory. For faster assertion checks, also pass
`-DCMAKE_CXX_FLAGS_DEBUG="-O2 -g" -DCMAKE_CUDA_FLAGS_DEBUG="-O2 -g"`;
these flags preserve assertions (no `NDEBUG`). The CPU reference regression checks per-head page selection
and a partial tail in both HND and NHD layouts.

```bash
python -m unittest evaluation.baseline.test_baseline evaluation.baseline.test_pg19 \
  evaluation.baseline.test_analysis evaluation.baseline.test_processes
python -m evaluation.baseline.freeze --output results/baseline/frozen-stable
```

Freeze writes environment/submodule records, source and binary snapshots,
`model-index.json` (SHA256 of weights/config/tokenizer), `artifact-index.json`
(exact input IDs, dataset/text hashes and result files), and archive checksums.
The model and experiment payloads are indexed rather than copied; preserve them
alongside the archive. See `docs/experiments/README.md` for the archive inventory.
