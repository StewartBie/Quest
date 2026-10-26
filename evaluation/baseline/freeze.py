"""Snapshot source, build, environment and artifact hashes without Git mutation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, default=Path("results/baseline/frozen"))
    p.add_argument("--model", type=Path, default=Path("data/models/longchat-7b-v1.5-32k"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def capture(name, command):
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (args.output / name).write_text(result.stdout)
    capture("requirements.txt", [sys.executable, "-m", "pip", "freeze"])
    capture("git-status.txt", ["git", "status", "--short"])
    capture("git-diff.patch", ["git", "diff", "--binary"])
    capture("submodules.txt", ["git", "submodule", "status", "--recursive"])
    capture("nvcc.txt", ["/home/bzq/.local/envs/quest/bin/nvcc", "--version"])
    capture("gpu.csv", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.sm,clocks.mem,temperature.gpu,power.draw", "--format=csv"])
    # Pin actual local weights and tokenizer, including downloads without revision metadata.
    model_files = [path for path in args.model.rglob("*") if path.is_file()
                   and ".cache" not in path.relative_to(args.model).parts]
    if not model_files:
        raise ValueError(f"No model files found at {args.model}")
    model_index = [dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))
                   for path in sorted(model_files)]
    (args.output / "model-index.json").write_text(json.dumps(model_index, indent=2)+"\n")
    files = set()
    for root in ("evaluation/baseline", "kernels/minimal", "kernels/include",
                 "kernels/3rdparty/flashinfer/include", "kernels/3rdparty/googletest",
                 "docs/reports", "docs/experiments"):
        for path in Path(root).rglob("*"):
            if path.is_file() and not any(part in ("__pycache__", ".git", ".pytest_cache") for part in path.parts):
                files.add(path)
    for file in ("WORK_SUMMARY_AND_PLAN.md", "pyproject.toml", ".gitmodules", "README.md", "kernels/src/test/test_max_possible.cu",
                 "kernels/src/include/cpu_reference.h", "kernels/src/include/cpu_utils.h",
                 "kernels/build-sm120-gcc13/CMakeCache.txt", "kernels/build-sm120-gcc13/libquest_probe.so",
                 "kernels/build-sm120-debug/CMakeCache.txt", "kernels/build-sm120-debug/libquest_probe.so",
                 "kernels/build-sm120-debug/test_max_possible", "kernels/build-sm120-debug/Testing/Temporary/LastTest.log"):
        files.add(Path(file))
    with zipfile.ZipFile(args.output / "source-build.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path))
    artifact_index = []
    for path in sorted(Path("results/baseline").rglob("*")):
        if path.is_file() and args.output not in path.parents:
            artifact_index.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path)))
    (args.output / "artifact-index.json").write_text(json.dumps(artifact_index, indent=2)+"\n")
    frozen_index = [dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))
                    for path in sorted(args.output.iterdir()) if path.is_file()]
    (args.output / "checksums.json").write_text(json.dumps(frozen_index, indent=2)+"\n")
    print(f"Frozen {len(files)} source/build files and indexed {len(artifact_index)} experiment artifacts")


if __name__ == "__main__":
    main()
