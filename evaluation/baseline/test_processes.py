"""Check that ordinary measurements never share a profiling interpreter."""
import argparse
from pathlib import Path
import unittest

from .run import benchmark_commands


class ProcessTests(unittest.TestCase):
    def test_profile_configuration_isolation(self):
        args = argparse.Namespace(model="model", seed=42, tail_tokens=64,
            decode_steps=8, repeats=3, profile_level="minimal", lengths=[4096, 8192],
            budgets=[0, 64], backends=["native", "split"], instrument=True, snapshot=True)
        commands = benchmark_commands(args, Path("results/test"))
        self.assertEqual(len(commands), 9)
        self.assertEqual(len({c[c.index("--output") + 1] for c in commands}), 9)
        self.assertIn("--instrument", commands[0])
        for command in commands:
            self.assertNotIn("--profile", command)
        for command in commands[1:]:
            self.assertEqual(command[command.index("--worker-mode") + 1], "profile")
            self.assertNotIn("--instrument", command)
            self.assertNotIn("--snapshot", command)
            for option in ("--lengths", "--budgets", "--backends"):
                self.assertTrue(command[command.index(option) + 2].startswith("--")
                                if command.index(option) + 2 < len(command) else True)


if __name__ == "__main__":
    unittest.main()
