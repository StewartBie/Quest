"""Validated FP16 MHA interface to the repository's original CUDA kernels."""
import ctypes
import functools
import weakref
from pathlib import Path

import torch


def on_input_device(method):
    @functools.wraps(method)
    def call(self, q, state, *args):
        self.validate(q, state)
        with torch.cuda.device(q.device):
            return method(self, q, state, *args)
    return call


def release_plans(lib, plans):
    # Do not retain NativeKernels here: the finalizer must not own its target.
    for key, handle in list(plans.items()):
        with torch.cuda.device(key[0]):
            lib.quest_release(handle)
        del plans[key]


class NativeKernels:
    def __init__(self, library=None):
        path = library or Path(__file__).resolve().parents[2] / "kernels/build-sm120-gcc13/libquest_probe.so"
        self.lib = ctypes.CDLL(str(path))
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        self.lib.quest_score.argtypes = [ptr] * 5 + [integer] * 4 + [ptr]
        self.lib.quest_decode.argtypes = [ptr] * 5 + [integer] * 6 + [ptr, ptr]
        self.lib.quest_plan.argtypes = [ctypes.POINTER(ptr)] + [integer] * 4 + [ptr]
        self.lib.quest_plan.restype = integer
        self.lib.quest_release.argtypes = [ptr]
        self.lib.quest_release.restype = None
        self.plans = {}
        self._finalizer = weakref.finalize(self, release_plans, self.lib, self.plans)
        self.lib.quest_score.restype = self.lib.quest_decode.restype = integer
        self.lib.quest_error.argtypes = [integer]
        self.lib.quest_error.restype = ctypes.c_char_p

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.quest_error(code).decode())

    @staticmethod
    def validate(q, state):
        if (not q.is_cuda or q.dtype != torch.float16 or not q.is_contiguous()
                or q.shape != (1, state.heads, state.dim)
                or q.device != state.data.device or state.dim not in (64, 128)):
            raise ValueError("Expected contiguous CUDA FP16 [1, heads, 64/128] on cache device")

    @on_input_device
    def score(self, q, state):
        pages = state.pages
        out = torch.empty((state.heads, pages - 1), dtype=q.dtype, device=q.device)
        if pages > 1:
            chunks = (pages + state.page_size - 1) // state.page_size
            indptr = state.indptr(chunks)
            self.check(self.lib.quest_score(q.data_ptr(), out.data_ptr(),
                state.metadata.data_ptr(), state.arange.data_ptr(), indptr.data_ptr(),
                state.heads, state.dim, state.page_size, pages,
                torch.cuda.current_stream(q.device).cuda_stream))
        return out

    @on_input_device
    def decode(self, q, state, selected):
        if (selected.dtype != torch.int32 or not selected.is_contiguous()
                or selected.ndim != 2 or selected.shape[0] != state.heads
                or selected.device != q.device or selected.shape[1] >= state.pages):
            raise ValueError("Expected int32 [heads, selected_pages_without_tail]")
        out = torch.empty_like(q)
        count = selected.shape[1]
        stream = torch.cuda.current_stream(q.device).cuda_stream
        key = (q.device.index, state.heads, state.dim, state.page_size, count, stream)
        if count and key not in self.plans:
            handle = ctypes.c_void_p()
            self.check(self.lib.quest_plan(ctypes.byref(handle), state.heads,
                state.dim, state.page_size, count, stream))
            self.plans[key] = handle
        # The original handler appends a separate partition for the last page.
        indptr = state.indptr(count if count else 1)
        self.check(self.lib.quest_decode(q.data_ptr(), out.data_ptr(), state.data.data_ptr(),
            selected.data_ptr(), indptr.data_ptr(), state.heads, state.dim, state.page_size,
            selected.shape[1], state.last_len, state.pages - 1,
            stream, self.plans.get(key)))
        return out

    def close(self):
        # Keep the finalizer active: callers may reuse this object after close.
        release_plans(self.lib, self.plans)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
