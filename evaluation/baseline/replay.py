"""Paired page-address/order experiment on a real-model Q/KV snapshot.

Same query and selected token set, with physical slots permuted consistently.
Warm CUDA-graph batches amortize Python launch overhead. No claims about DRAM
bandwidth: request-derived payload/time is not measured DRAM traffic.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .attention import PageState
from .native import NativeKernels


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("snapshot", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--profile-one", action="store_true", help="One warmed kernel per variant for external ncu")
    args = p.parse_args()
    payload = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    q, data = payload["q"].cuda(), payload["data"].cuda()
    selected = payload["selected"].cuda()
    state = PageState(q.shape[1], q.shape[2], payload["length"], payload["page_size"], q.device)
    state.length = payload["length"]
    state.data.copy_(data)
    state.metadata.copy_(payload["metadata"][:state.metadata.shape[0]].cuda())
    native = NativeKernels()
    expected = native.decode(q, state, selected).clone()
    generator = torch.Generator(device="cuda").manual_seed(42)
    permutation = torch.cat((torch.randperm(state.pages-1, device="cuda", generator=generator),
                             torch.tensor([state.pages-1], device="cuda")))
    relocated = torch.empty_like(data)
    relocated[permutation] = data
    output = []
    for layout in ("original", "permuted_physical_slots"):
        state.data = data if layout == "original" else relocated
        indices = selected if layout == "original" else permutation[selected.long()].int()
        for order in ("topk_order", "sorted_physical"):
            chosen = indices if order == "topk_order" else indices.sort(dim=-1).values
            actual = native.decode(q, state, chosen)
            torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.005)
            if args.profile_one:
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(f"replay/{layout}/{order}")
                native.decode(q, state, chosen)
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                continue
            def fused():
                return native.decode(q, state, chosen)
            def gather():
                return state.gather(chosen)
            gathered_k, gathered_v = gather()
            def attention():
                return F.scaled_dot_product_attention(q.unsqueeze(2), gathered_k, gathered_v)
            for name, fn in (("fused", fused), ("gather_diagnostic", gather), ("sdpa_diagnostic", attention)):
                # Decode plans are keyed by stream. Warm on the SAME stream as
                # capture: planning inside capture would record H2D copies from
                # temporary host vectors whose storage dies before graph replay.
                capture_stream = torch.cuda.Stream()
                capture_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(capture_stream):
                    for _ in range(3):
                        fn()
                capture_stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    for _ in range(args.batch):
                        result = fn()
                graph.replay()
                torch.cuda.synchronize()
                if name == "fused":
                    torch.testing.assert_close(result, expected, atol=0.005, rtol=0.005)
                samples = []
                for _ in range(args.repeats):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000 / args.batch)
                bytes_requested = 2 * state.heads * (chosen.shape[1] * state.page_size + state.last_len) * state.dim * 2
                row = dict(layout=layout, order=order, operation=name, us=samples,
                    median_us=float(np.median(samples)), mean_us=float(np.mean(samples)),
                    requested_kv_bytes=bytes_requested,
                    request_payload_gb_s=bytes_requested / (float(np.median(samples)) * 1000),
                    cache_condition="warm repeated identical Q/KV; no cache flush")
                output.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "us"}), flush=True)
                del graph, result
    args.output.write_text(json.dumps(dict(snapshot=str(args.snapshot), results=output,
        warning="Paired microbenchmark only; payload throughput is NOT DRAM bandwidth; split times do not decompose fused time"), indent=2)+"\n")
    native.close()


if __name__ == "__main__":
    main()
