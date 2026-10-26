"""Small numerical checks before using the adapted pipeline for research."""
import unittest
import gc
import weakref
from unittest import mock

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from .attention import NativeKernels, PageState, PagedCache, Recorder, install


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class BaselineTests(unittest.TestCase):
    @torch.inference_mode()
    def test_plan_lifecycle_and_device_guard(self):
        original = torch.cuda.current_device()
        device = torch.device("cuda", 1 if torch.cuda.device_count() > 1 else original)
        k = torch.randn(1, 4, 49, 64, device=device, dtype=torch.float16)
        v = torch.randn_like(k)
        state = PageState(4, 64, 49, 16, device)
        state.append(k, v)
        q = torch.randn(1, 4, 64, device=device, dtype=torch.float16)
        native = NativeKernels()
        releases = []
        release = native.lib.quest_release
        def tracked_release(handle):
            releases.append((handle.value, torch.cuda.current_device()))
            release(handle)
        with mock.patch.object(native.lib, "quest_release", side_effect=tracked_release):
            selected = native.score(q, state).topk(2, dim=-1).indices.int()
            actual = native.decode(q, state, selected)
            keys, values = state.gather(selected)
            expected = F.scaled_dot_product_attention(q.unsqueeze(2), keys, values).squeeze(2)
            torch.cuda.synchronize(device)
            torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.005)
            self.assertEqual(torch.cuda.current_device(), original)
            self.assertEqual(len(native.plans), 1)
            native.close()
            native.close()
            self.assertEqual(len(releases), 1)
            # Reuse after explicit close, then exercise automatic destruction.
            native.decode(q, state, selected)
            ref = weakref.ref(native)
            del native
            gc.collect()
            self.assertIsNone(ref())
            self.assertEqual(len(releases), 2)
            self.assertTrue(all(index == device.index for _, index in releases))
            self.assertEqual(torch.cuda.current_device(), original)

    @unittest.skipUnless(torch.cuda.device_count() > 1, "Two CUDA devices required")
    def test_cross_device_error_restores_context(self):
        original = torch.cuda.current_device()
        with NativeKernels() as native:
            state = PageState(4, 64, 49, 16, torch.device("cuda", 1))
            q = torch.zeros(1, 4, 64, device="cuda:1", dtype=torch.float16)
            bad = torch.zeros(4, 1, device="cuda:1", dtype=torch.int64)
            with self.assertRaises(ValueError):
                native.decode(q, state, bad)
        self.assertEqual(torch.cuda.current_device(), original)

    @torch.inference_mode()
    def test_decode_cuda_graph_with_prebuilt_stream_plan(self):
        torch.manual_seed(19)
        native = NativeKernels()
        k = torch.randn(1, 4, 257, 128, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        state = PageState(4, 128, 257, 16, k.device)
        state.append(k, v)
        q = torch.randn(1, 4, 128, device="cuda", dtype=torch.float16)
        selected = native.score(q, state).topk(3, dim=-1).indices.int()
        expected = native.decode(q, state, selected).clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            native.decode(q, state, selected)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(4):
                actual = native.decode(q, state, selected)
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.005)
        del graph
        native.close()

    @torch.inference_mode()
    def test_score_decode_and_tail(self):
        torch.manual_seed(42)
        native = NativeKernels()
        # Cross metadata-chunk and KV-page boundaries, including an exact page.
        for length in (17, 31, 32, 255, 256, 257, 1101):
            h, d, p = 4, 128, 16
            k = torch.randn(1, h, length, d, device="cuda", dtype=torch.float16)
            v = torch.randn_like(k)
            state = PageState(h, d, length + 1, p, k.device)
            state.append(k[:, :, :-1], v[:, :, :-1])
            state.append(k[:, :, -1:], v[:, :, -1:])
            q = torch.randn(1, h, d, device="cuda", dtype=torch.float16)
            scores = native.score(q, state)
            keys = k[0, :, :(state.pages-1)*p].reshape(h, state.pages-1, p, d).float()
            ref = torch.maximum(q[0, :, None].float() * keys.amax(2),
                                q[0, :, None].float() * keys.amin(2)).sum(-1)
            torch.testing.assert_close(scores.float(), ref, atol=0.15, rtol=0.002)
            for budget in (1, min(3, state.pages), state.pages):
                selected = scores.topk(budget - 1, dim=-1, sorted=False).indices.int()
                gathered_k, gathered_v = state.gather(selected)
                expected = F.scaled_dot_product_attention(q.unsqueeze(2), gathered_k, gathered_v).squeeze(2)
                actual = native.decode(q, state, selected)
                torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.005)
                if budget == state.pages:
                    full = F.scaled_dot_product_attention(q.unsqueeze(2), k, v).squeeze(2)
                    torch.testing.assert_close(actual, full, atol=0.005, rtol=0.005)

    @torch.inference_mode()
    def test_model_full_budget_matches_huggingface(self):
        torch.manual_seed(7)
        config = LlamaConfig(vocab_size=128, hidden_size=256, intermediate_size=512,
                            num_hidden_layers=3, num_attention_heads=4,
                            num_key_value_heads=4, max_position_embeddings=512)
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM(config).half().cuda().eval()
        ids = torch.randint(0, 128, (1, 65), device="cuda")
        expected = []
        cache = None
        for tokens in (ids[:, :62], ids[:, 62:63], ids[:, 63:64], ids[:, 64:]):
            out = model(tokens, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            expected.append(out.logits[:, -1].clone())
        install(model, Recorder(), page_budget=0)
        cache = PagedCache(80)
        for i, tokens in enumerate((ids[:, :62], ids[:, 62:63], ids[:, 63:64], ids[:, 64:])):
            actual = model(tokens, past_key_values=cache, use_cache=True).logits[:, -1]
            torch.testing.assert_close(actual, expected[i], atol=0.003, rtol=0.02)
        # Reinitializing cache must reset positions, metadata, and layer state.
        actual = model(ids[:, :62], past_key_values=PagedCache(80)).logits[:, -1]
        torch.testing.assert_close(actual, expected[0], atol=0.003, rtol=0.02)


if __name__ == "__main__":
    unittest.main()
