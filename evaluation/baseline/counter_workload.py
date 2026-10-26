"""One warmed, numerically checked operation inside an NCU NVTX filter."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .attention import PageState
from .native import NativeKernels


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("snapshot", type=Path)
    p.add_argument("--stage", choices=("score", "topk", "fused", "gather", "sdpa"), required=True)
    p.add_argument("--layout", choices=("original", "permuted"), default="original")
    p.add_argument("--order", choices=("topk", "sorted"), default="topk")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    payload = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    q, selected = payload["q"].cuda(), payload["selected"].cuda()
    state = PageState(q.shape[1], q.shape[2], payload["length"], payload["page_size"], q.device)
    state.length = payload["length"]
    state.data.copy_(payload["data"].cuda())
    state.metadata.copy_(payload["metadata"][:state.metadata.shape[0]].cuda())
    native = NativeKernels()
    scores = native.score(q, state)
    selected_k, selected_v = state.gather(selected)
    reference = F.scaled_dot_product_attention(q.unsqueeze(2), selected_k, selected_v).squeeze(2)
    torch.testing.assert_close(native.decode(q, state, selected), reference, atol=0.005, rtol=0.005)
    if args.layout == "permuted":
        generator = torch.Generator(device="cuda").manual_seed(42)
        permutation = torch.cat((torch.randperm(state.pages - 1, device="cuda", generator=generator),
                                 torch.tensor([state.pages - 1], device="cuda")))
        relocated = torch.empty_like(state.data)
        relocated[permutation] = state.data
        state.data = relocated
        selected = permutation[selected.long()].int()
    if args.order == "sorted":
        selected = selected.sort(dim=-1).values
    selected_k, selected_v = state.gather(selected)
    torch.testing.assert_close(native.decode(q, state, selected), reference, atol=0.005, rtol=0.005)
    operations = {
        "score": lambda: native.score(q, state),
        "topk": lambda: torch.topk(scores, selected.shape[1], dim=-1, sorted=False).indices.int(),
        "fused": lambda: native.decode(q, state, selected),
        "gather": lambda: state.gather(selected),
        "sdpa": lambda: F.scaled_dot_product_attention(q.unsqueeze(2), selected_k, selected_v),
    }
    fn = operations[args.stage]
    # All allocation, planning, reference checks and warmup are outside capture.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("quest_capture")
    try:
        actual = fn()
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    if args.stage == "fused":
        torch.testing.assert_close(actual, reference, atol=0.005, rtol=0.005)
    elif args.stage == "score":
        torch.testing.assert_close(actual, scores)
    elif args.stage == "topk":
        expected_values = scores.topk(selected.shape[1], dim=-1).values
        actual_values = scores.gather(1, actual.long()).sort(dim=-1, descending=True).values
        torch.testing.assert_close(actual_values, expected_values)
    elif args.stage == "gather":
        torch.testing.assert_close(actual[0], selected_k)
        torch.testing.assert_close(actual[1], selected_v)
    else:
        torch.testing.assert_close(actual.squeeze(2), reference, atol=0.005, rtol=0.005)
    args.output.write_text(json.dumps(dict(stage=args.stage, layout=args.layout, order=args.order,
        snapshot=str(args.snapshot), layer=payload["layer"], step=payload["step"],
        length=state.length, page_size=state.page_size, selected_pages=selected.shape[1]+1,
        query_heads=state.heads, numerical_check="passed", nvtx_range="quest_capture",
        caveat="Standalone snapshot; gather+SDPA does not decompose fused attention"), indent=2)+"\n")
    native.close()


if __name__ == "__main__":
    main()
