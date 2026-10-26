#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONDONTWRITEBYTECODE=1
QUEST_PYTHON="${QUEST_PYTHON:-/home/bzq/.local/envs/quest/bin/python}"
exec "$QUEST_PYTHON" -m evaluation.baseline.counters "$@"
