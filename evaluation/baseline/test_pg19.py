"""Verify target alignment, dense equivalence, sparse selection and aggregation."""
import contextlib
import io
import math
import unittest

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from .attention import PagedCache, Recorder, install
from .pg19 import context_limit, metrics, score_document, summarize
from .run import forward_tokens


class AggregationTests(unittest.TestCase):
    def test_token_weighting_and_paired_inputs(self):
        rows = []
        for budget in (0, 16):
            for index, (count, nll) in enumerate(((1, 1.0), (3, 9.0))):
                rows.append(dict(document_index=index, input_sha256=str(index),
                                 token_budget=budget, **metrics(nll, count)))
        result = summarize(rows)
        self.assertAlmostEqual(result[0]["perplexity"], math.exp(2.5))
        self.assertEqual(result[1]["delta_ppl"], 0)
        rows[-1]["input_sha256"] = "different"
        with self.assertRaises(ValueError):
            summarize(rows)

    def test_invalid_counts_and_context_limit(self):
        with self.assertRaises(ValueError):
            metrics(0, 0)
        with self.assertRaises(ValueError):
            metrics(float("nan"), 1)
        config = LlamaConfig(max_position_embeddings=4096,
                             rope_scaling={"type": "linear", "factor": 8.0})
        self.assertEqual(context_limit(config), 32768)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ScoringTests(unittest.TestCase):
    @torch.inference_mode()
    def test_dense_next_token_nll_and_sparse_against_split(self):
        torch.manual_seed(104)
        config = LlamaConfig(vocab_size=128, hidden_size=256, intermediate_size=384,
                            num_hidden_layers=3, num_attention_heads=4,
                            num_key_value_heads=4, max_position_embeddings=128)
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM(config).half().cuda().eval()
        ids = torch.randint(0, 128, (1, 50), device="cuda")
        # Independent teacher-forced causal reference for every next-token target.
        logits = model(ids[:, :-1], use_cache=False).logits.float()
        expected = F.cross_entropy(logits[0], ids[0, 1:], reduction="none").cpu()
        rows = []
        with contextlib.redirect_stdout(io.StringIO()):
            full = score_document(model, ids, 0, rows.append, log_every=17)
        self.assertEqual([r["target_position"] for r in rows], list(range(1, 50)))
        self.assertEqual(full["scored_tokens"], 49)
        self.assertAlmostEqual(full["total_nll"], sum(r["nll"] for r in rows))
        torch.testing.assert_close(torch.tensor([r["nll"] for r in rows]), expected,
                                   atol=0.003, rtol=0.002)
        # Large budget must be the same dense path, with a fresh document cache.
        same = []
        with contextlib.redirect_stdout(io.StringIO()):
            score_document(model, ids, 64, same.append, log_every=17)
        self.assertEqual([r["nll"] for r in same], [r["nll"] for r in rows])
        sparse = []
        with contextlib.redirect_stdout(io.StringIO()):
            score_document(model, ids, 16, sparse.append, log_every=17)
        # Independent gather+SDPA backend verifies a genuinely sparse run.
        install(model, Recorder(), page_budget=1, backend="split")
        cache = PagedCache(49)
        expected_sparse = []
        for t in range(49):
            output = forward_tokens(model, cache, ids[:, t:t+1])
            expected_sparse.append(F.cross_entropy(output[:, -1], ids[:, t+1]).item())
        torch.testing.assert_close(torch.tensor([r["nll"] for r in sparse]),
                                   torch.tensor(expected_sparse), atol=0.003, rtol=0.002)
        self.assertFalse(model.probe_native.plans)
        model.probe_native.close()


if __name__ == "__main__":
    unittest.main()
