#!/usr/bin/env bash
# Back up a generation run to a Hugging Face dataset repo as Parquet shards, once its worker exits.
#   nohup deploy/hf_backup.sh RUN REPO_ID [README] > runs/RUN/hf_backup.log 2>&1 &
# README (optional) replaces the dataset card, e.g. one that lists the new config.
set -euo pipefail
RUN="${1:?run name}"
REPO="${2:?repo id}"
README="${3:-}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
STAGE="${STAGE:-/workspace/hf_dataset}"

for pidfile in "$PROJECT_ROOT/runs/$RUN"/worker_gpu*.pid; do
    [[ -f "$pidfile" ]] || continue
    while kill -0 "$(cat "$pidfile")" 2>/dev/null; do sleep 60; done
done
echo "[hf_backup $(date +%H:%M:%S)] $RUN generation finished; packing"
"$PYTHON_BIN" -u "$PROJECT_ROOT/deploy/pack_dataset.py" pack "$DATA_ROOT/$RUN" --run-dir "$PROJECT_ROOT/runs/$RUN" \
    --out "$STAGE" --name "$RUN"
[[ -n "$README" ]] && cp "$README" "$STAGE/README.md"
# Re-running the folder upload skips files already on the Hub.
hf upload "$REPO" "$STAGE" . --repo-type dataset --commit-message "Add $RUN run"
echo "[hf_backup $(date +%H:%M:%S)] uploaded $RUN to $REPO"
