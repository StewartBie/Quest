"""Check profiler attribution/overlap math with independently specified events."""
import json
from pathlib import Path
import tempfile
import unittest

from .analyze import profile_summary
from .counters import summarize_csv, collection_error


class AnalysisTests(unittest.TestCase):
    def test_collection_error_shows_profiler_diagnostic(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "capture.log"
            path.write_text("==PROF== Connected\n==ERROR== InterprocessLockFailed\n")
            message = collection_error("score", path)
        self.assertIn("InterprocessLockFailed", message)
        self.assertIn("capture.log", message)

    def test_ncu_csv_preserves_kernel_identity_and_units(self):
        raw = '''==PROF== Disconnected
"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"1","decode","dram__bytes_read.sum","byte","1,024"
"1","decode","smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct","%","35.5"
"2","merge","gpu__time_duration.sum","nsecond","980"
'''
        result = summarize_csv(raw)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["Metric Value"], "1,024")
        self.assertEqual(result[1]["Metric Unit"], "%")
        self.assertEqual(result[2]["Kernel Name"], "merge")
        with self.assertRaises(ValueError):
            summarize_csv("==WARNING== No kernels were profiled")

    def test_kernel_invocations_and_gpu_union(self):
        events = [
            dict(ph="X", name="quest/layer_2/step_0/topk", ts=0, dur=10),
            dict(cat="cuda_runtime", name="cudaLaunchKernel", ts=1, dur=1, args=dict(correlation=1)),
            dict(cat="cuda_runtime", name="cudaLaunchKernel", ts=3, dur=1, args=dict(correlation=2)),
            dict(ph="X", name="quest/layer_2/step_1/topk", ts=20, dur=10),
            dict(cat="cuda_runtime", name="cudaLaunchKernel", ts=21, dur=1, args=dict(correlation=3)),
            dict(cat="kernel", name="select", ts=100, dur=5, args=dict(correlation=1)),
            dict(cat="kernel", name="cast", ts=106, dur=2, args=dict(correlation=2)),
            dict(cat="kernel", name="select", ts=110, dur=4, args=dict(correlation=3)),
            # An overlapping transfer must not be double counted as busy time.
            dict(cat="gpu_memcpy", name="Memcpy", ts=102, dur=5),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.json"
            path.write_text(json.dumps(dict(traceEvents=events)))
            result = profile_summary(path)
        self.assertEqual(result["gpu_span_us"], 14)
        self.assertEqual(result["gpu_work_union_us"], 12)
        self.assertEqual(result["gpu_gap_us"], 2)
        self.assertEqual(result["stage_kernel_us"]["topk"]["n"], 3)
        self.assertEqual(result["stage_invocation_kernel_us"]["topk"]["n"], 2)
        self.assertEqual(result["stage_invocation_kernel_us"]["topk"]["mean"], 5.5)


if __name__ == "__main__":
    unittest.main()
