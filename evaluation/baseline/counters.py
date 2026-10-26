"""Collect NCU reports as administrator, export CSV and retain metric units."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def summarize_csv(text):
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if "Metric Name" in line and "Metric Value" in line), None)
    if header is None:
        raise ValueError("NCU export has no metric header; no kernels may have matched the NVTX filter")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))
    metrics = []
    for row in rows:
        name = row.get("Metric Name", "")
        if (name.startswith(("dram__", "lts__", "sm__", "smsp__", "gpu__time", "launch__"))
                and any(word in name for word in ("throughput", "bytes", "hit_rate", "sector", "warps_active",
                                                  "occupancy", "stall", "duration", "register", "shared_mem"))):
            metrics.append({k: row.get(k, "") for k in ("ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value")})
    if not rows or not metrics:
        raise ValueError("No expected hardware metrics in NCU report")
    return metrics


def collection_error(name, log_path):
    text = log_path.read_text(errors="replace")
    errors = [line for line in text.splitlines() if "==ERROR==" in line]
    detail = "\n".join(errors[-8:]) or "\n".join(text.splitlines()[-12:])
    return f"Collection failed ({name}): {log_path}\n{detail}"


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--snapshot", type=Path, default=Path("results/baseline/sweep/snapshot-8192-64-native.pt"))
    p.add_argument("--output", type=Path)
    p.add_argument("--ncu", type=Path, default=Path("/home/bzq/.local/envs/quest/nsight-compute-2025.1.1/ncu"))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cache-control", choices=("all", "none"), default="all")
    p.add_argument("--set", dest="metric_set", choices=("full", "basic"), default="full")
    p.add_argument("--include-split", action="store_true", help="Also collect independent gather and SDPA diagnostics")
    p.add_argument("--dry-run", action="store_true", help="Print commands; do not use sudo or GPU")
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    snapshot, ncu = args.snapshot.resolve(), args.ncu.resolve()
    library = repo / "kernels/build-sm120-gcc13/libquest_probe.so"
    for path in (snapshot, ncu, library):
        if not path.is_file():
            p.error(f"Missing file: {path}")
    if args.gpu < 0:
        p.error("GPU must be nonnegative")
    output = (args.output or Path("results/baseline") / ("counters-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))).resolve()
    # Stable across this script's runs/users so NCU still serializes captures.
    # A non-sticky directory avoids protected_regular on an old /tmp lock file.
    temporary = repo / "results/baseline/ncu-tmp"
    if output.exists():
        p.error(f"Output already exists; choose a new directory: {output}")
    variants = [("score", "original", "topk"), ("topk", "original", "topk")]
    for stage in (["fused", "gather", "sdpa"] if args.include_split else ["fused"]):
        variants += [(stage, layout, order) for layout in ("original", "permuted") for order in ("topk", "sorted")]
    prefix = [] if os.geteuid() == 0 else ["sudo"]
    commands = []
    for stage, layout, order in variants:
        name = f"{stage}-{layout}-{order}"
        command = prefix + ["env", f"CUDA_VISIBLE_DEVICES={args.gpu}", "PYTHONDONTWRITEBYTECODE=1",
            f"TMPDIR={temporary}",
            str(ncu), "--config-file", "0", "--set", args.metric_set,
            "--nvtx", "--nvtx-include", "quest_capture/", "--cache-control", args.cache_control,
            "--clock-control", "none", "--target-processes", "application-only",
            "--export", str(output / name), sys.executable, "-m", "evaluation.baseline.counter_workload",
            str(snapshot), "--stage", stage, "--layout", layout, "--order", order,
            "--output", str(output / f"{name}.workload.json")]
        commands.append((name, command))
    if args.dry_run:
        for _, command in commands:
            print(shlex.join(command))
        return
    output.mkdir(parents=True)
    temporary.mkdir(parents=True, mode=0o700, exist_ok=True)
    if os.geteuid() == 0 and "SUDO_UID" in os.environ and "SUDO_GID" in os.environ:
        os.chown(temporary, int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"]))
    manifest = dict(snapshot=str(snapshot), snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(), python=sys.executable,
        ncu=subprocess.check_output([str(ncu), "--version"], text=True), gpu=args.gpu,
        cache_control=args.cache_control, clock_control="none", metric_set=args.metric_set,
        profiler_temporary_directory=str(temporary),
        commands=[command for _, command in commands],
        caveats=["Kernel replay alters cache state; profiled latency is not model baseline latency",
                 "cache-control all flushes caches between replay passes; none does not guarantee in-model cache residency",
                 "NVTX filters exclude planning, reference checks and warmup; fused includes decode and merge kernels",
                 "Raw NCU metric units are preserved; requested payload/time is not substituted for DRAM traffic"])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    completed = []
    try:
        if prefix:
            subprocess.run(["sudo", "-v"], check=True)
        for name, command in commands:
            print(f"Collecting {name} ...", flush=True)
            with (output / f"{name}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            report = output / f"{name}.ncu-rep"
            if result.returncode or not report.exists():
                raise RuntimeError(collection_error(name, output / f"{name}.log"))
            exported = subprocess.run([str(ncu), "--config-file", "0", "--import", str(report),
                "--page", "raw", "--csv"], text=True, capture_output=True, check=True)
            (output / f"{name}.csv").write_text(exported.stdout)
            (output / f"{name}.export.log").write_text(exported.stderr)
            metrics = summarize_csv(exported.stdout)
            completed.append(dict(variant=name, metrics=metrics))
            (output / "summary.json").write_text(json.dumps(completed, indent=2)+"\n")
        (output / "SUCCESS").write_text("All requested variants collected and exported.\n")
        print(f"Completed: {output}\nRead summary.json; retain *.ncu-rep and *.csv", flush=True)
    finally:
        # When called with sudo bash, return only this newly-created result tree
        # to the invoking user's ownership. No driver configuration is changed.
        if os.geteuid() == 0 and "SUDO_UID" in os.environ and "SUDO_GID" in os.environ:
            uid, gid = int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"])
            for path in output.rglob("*"):
                if not path.is_symlink():
                    os.chown(path, uid, gid)
            os.chown(output, uid, gid)


if __name__ == "__main__":
    main()
