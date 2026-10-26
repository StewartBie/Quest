"""Instrumentable Quest adaptation for transformers==4.45.2, batch=1 MHA.

Original CUDA score and partitioned paged decode, torch.topk, HND storage,
SDPA prefill. Last page always included, first two layers use full attention.
The split diagnostic is gather+SDPA; its times do NOT decompose fused decode.
"""
import contextlib
import json
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from .native import NativeKernels


class Recorder:
    def __init__(self, mode="none"):
        self.mode = mode  # none, timing, trace, profile
        self.step = -1
        self.phase = "prefill"
        self.events = []
        self.pages = []
        self.context = {}
        self.capture_snapshot = False
        self.snapshot = None

    @contextlib.contextmanager
    def stage(self, layer, name):
        active = self.step >= 0 and self.mode in ("timing", "profile")
        if not active:
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        label = f"quest/layer_{layer}/step_{self.step}/{name}"
        torch.cuda.nvtx.range_push(label)
        begin = time.perf_counter_ns()
        start.record()
        try:
            with torch.profiler.record_function(label):
                yield
        finally:
            end.record()
            host_us = (time.perf_counter_ns() - begin) / 1000
            torch.cuda.nvtx.range_pop()
            self.events.append((dict(self.context, layer=layer, step=self.step,
                phase=self.phase, stage=name, host_enqueue_us=host_us), start, end))

    def selection(self, layer, state, selected, sparse):
        if self.mode != "trace" or self.step < 0:
            return
        # No .cpu()/.tolist() inside the measured execution. Separate trace run.
        self.pages.append((dict(self.context, layer=layer, step=self.step,
            phase=self.phase, sequence_length=state.length, page_size=state.page_size,
            last_page=state.pages - 1, last_page_length=state.last_len,
            cache_base_address=state.data.data_ptr(),
            page_stride_bytes=state.data.stride(0) * state.data.element_size(),
            head_stride_bytes=state.data.stride(2) * state.data.element_size(),
            value_offset_bytes=state.data.stride(1) * state.data.element_size(),
            sparse=sparse), selected.detach().clone()))

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        torch.cuda.synchronize()
        if self.events:
            with (folder / "stages.jsonl").open("a") as f:
                for row, start, end in self.events:
                    f.write(json.dumps(dict(row, gpu_interval_us=start.elapsed_time(end) * 1000)) + "\n")
        if self.pages:
            with (folder / "pages.jsonl").open("a") as f:
                for row, selection in self.pages:
                    for head, pages in enumerate(selection.cpu().tolist()):
                        # Logical and physical slots are identical in this allocator.
                        f.write(json.dumps(dict(row, query_head=head, kv_head=head,
                            logical_pages=pages + [row["last_page"]],
                            physical_pages=pages + [row["last_page"]])) + "\n")
        self.events.clear()
        self.pages.clear()
        if self.snapshot is not None:
            torch.save(self.snapshot, folder / ("snapshot-" + "-".join(str(self.context[k])
                       for k in ("context_length", "page_budget", "backend")) + ".pt"))
            self.snapshot = None


class PageState:
    def __init__(self, heads, dim, capacity, page_size, device):
        self.heads, self.dim, self.page_size = heads, dim, page_size
        self.capacity = (capacity + page_size - 1) // page_size
        self.data = torch.empty((self.capacity, 2, heads, page_size, dim),
                                device=device, dtype=torch.float16)
        chunks = (self.capacity + page_size - 1) // page_size
        self.metadata = torch.empty((chunks, 2, heads, page_size, dim),
                                    device=device, dtype=torch.float16)
        self.arange = torch.arange(max(chunks, self.capacity), device=device, dtype=torch.int32)
        self.head_ids = torch.arange(heads, device=device)[:, None]
        self._indptr = {}
        self.length = 0

    @property
    def pages(self):
        return (self.length + self.page_size - 1) // self.page_size

    @property
    def last_len(self):
        return (self.length - 1) % self.page_size + 1

    def indptr(self, count):
        if count not in self._indptr:
            self._indptr[count] = torch.tensor([0, count], device=self.data.device, dtype=torch.int32)
        return self._indptr[count]

    def append(self, k, v):
        # k/v: [1, H, T, D]. Initial prefill or single token, never chunked prefill.
        n = k.shape[-2]
        if self.length + n > self.capacity * self.page_size:
            raise ValueError("KV cache capacity exceeded")
        if self.length == 0:
            self.length = n
            padded = self.pages * self.page_size - n
            for channel, tensor in enumerate((k, v)):
                blocks = F.pad(tensor[0], (0, 0, 0, padded)).reshape(
                    self.heads, self.pages, self.page_size, self.dim).permute(1, 0, 2, 3)
                self.data[:self.pages, channel].copy_(blocks)
            # Reduce valid full pages and the valid part of the tail separately.
            for start in range(0, self.pages, self.page_size):
                stop = min(start + self.page_size, self.pages)
                keys = self.data[start:stop, 0]
                maximum, minimum = keys.amax(2), keys.amin(2)
                if stop == self.pages and padded:
                    maximum[-1] = keys[-1, :, :self.last_len].amax(1)
                    minimum[-1] = keys[-1, :, :self.last_len].amin(1)
                self.metadata[start // self.page_size, 0, :, :stop-start].copy_(maximum.transpose(0, 1))
                self.metadata[start // self.page_size, 1, :, :stop-start].copy_(minimum.transpose(0, 1))
        else:
            if n != 1:
                raise ValueError("Only initial prefill and single-token decode are supported")
            page, slot = divmod(self.length, self.page_size)
            self.data[page, 0, :, slot].copy_(k[0, :, 0])
            self.data[page, 1, :, slot].copy_(v[0, :, 0])
            chunk, entry = divmod(page, self.page_size)
            maximum, minimum = self.metadata[chunk, 0, :, entry], self.metadata[chunk, 1, :, entry]
            if slot == 0:
                maximum.copy_(k[0, :, 0])
                minimum.copy_(k[0, :, 0])
            else:
                torch.maximum(maximum, k[0, :, 0], out=maximum)
                torch.minimum(minimum, k[0, :, 0], out=minimum)
            self.length += 1

    def gather(self, selected):
        pages = torch.cat((selected.long(), torch.full((self.heads, 1), self.pages - 1,
                         device=self.data.device, dtype=torch.long)), dim=1)
        # Advanced indexing gathers only the selected head's KV bytes.
        k = self.data[pages, 0, self.head_ids].flatten(1, 2)
        v = self.data[pages, 1, self.head_ids].flatten(1, 2)
        valid_length = selected.shape[1] * self.page_size + self.last_len
        return k[:, :valid_length].unsqueeze(0), v[:, :valid_length].unsqueeze(0)


class PagedCache(Cache):
    def __init__(self, capacity, page_size=16):
        super().__init__()
        self.capacity, self.page_size = capacity, page_size
        self.states = {}

    def get_seq_length(self, layer_idx=0):
        return self.states[layer_idx].length if layer_idx in self.states else 0

    def get_max_cache_shape(self):
        return None

    def get_max_length(self):
        return None

    def update(self, *args, **kwargs):
        raise NotImplementedError("Use patched attention with PagedCache")


def forward(self, hidden_states, attention_mask=None, position_ids=None,
            past_key_value=None, output_attentions=False, use_cache=True,
            cache_position=None, position_embeddings=None, **kwargs):
    b, n, _ = hidden_states.shape
    if b != 1 or output_attentions or not isinstance(past_key_value, PagedCache):
        raise ValueError("Baseline requires batch=1, PagedCache, output_attentions=False")
    if attention_mask is not None:
        raise ValueError("Padding/custom masks are not supported by this baseline")
    if self.num_heads != self.num_key_value_heads or self.config.pretraining_tp != 1:
        raise ValueError("Baseline supports MHA and pretraining_tp=1 only")
    rec = self.probe_recorder
    layer = self.layer_idx
    with rec.stage(layer, "projection_rope"):
        q = self.q_proj(hidden_states).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings if position_embeddings is not None else self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    with rec.stage(layer, "kv_append_metadata"):
        if layer not in past_key_value.states:
            past_key_value.states[layer] = PageState(self.num_heads, self.head_dim,
                past_key_value.capacity, past_key_value.page_size, hidden_states.device)
        state = past_key_value.states[layer]
        was_empty = state.length == 0
        state.append(k, v)
    if was_empty:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=n > 1, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(b, n, -1)
    else:
        q = q.transpose(1, 2).reshape(1, self.num_heads, self.head_dim).contiguous()
        budget = state.pages if layer < 2 or self.probe_budget == 0 else min(state.pages, self.probe_budget)
        sparse = budget < state.pages
        if sparse:
            with rec.stage(layer, "page_score"):
                scores = self.probe_native.score(q, state)
            with rec.stage(layer, "topk"):
                selected = torch.topk(scores, budget - 1, dim=-1, sorted=False).indices.to(torch.int32)
        else:
            with rec.stage(layer, "full_page_indices"):
                selected = state.arange[:state.pages - 1].unsqueeze(0).expand(state.heads, -1).contiguous()
        rec.selection(layer, state, selected, sparse)
        if rec.capture_snapshot and layer == 16 and rec.step == 0:
            # Snapshot collection is restricted to the separate trace run.
            rec.snapshot = dict(q=q.detach().cpu(), data=state.data[:state.pages].detach().cpu(),
                metadata=state.metadata.detach().cpu(), selected=selected.detach().cpu(),
                length=state.length, page_size=state.page_size, layer=layer, step=rec.step,
                context=rec.context.copy())
        if self.probe_backend == "native":
            with rec.stage(layer, "kv_read_attention_fused"):
                out = self.probe_native.decode(q, state, selected)
            out = out.reshape(1, 1, -1)
        else:
            with rec.stage(layer, "kv_gather_diagnostic"):
                gathered_k, gathered_v = state.gather(selected)
            with rec.stage(layer, "attention_sdpa_diagnostic"):
                out = F.scaled_dot_product_attention(q.unsqueeze(2), gathered_k, gathered_v,
                    is_causal=False, dropout_p=0.0).reshape(1, 1, -1)
    with rec.stage(layer, "output_projection"):
        out = self.o_proj(out)
    return out, None, past_key_value


def install(model, recorder, page_budget=64, backend="native", library=None):
    if backend not in ("native", "split") or page_budget < 0:
        raise ValueError("Invalid backend/budget")
    if not hasattr(model, "probe_native"):
        model.probe_native = NativeKernels(library)
    native = model.probe_native
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.probe_recorder, attn.probe_budget = recorder, page_budget
        attn.probe_native, attn.probe_backend = native, backend
        attn.forward = types.MethodType(forward, attn)
