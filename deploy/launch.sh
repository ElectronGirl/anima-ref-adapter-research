#!/usr/bin/env bash
# Start one worker per GPU in the background.
#   deploy/launch.sh RUN_NAME [smoke|production]
set -euo pipefail
RUN="${1:?run name}"
MODE="${2:-production}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="$PROJECT_ROOT/runs/$RUN"
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
SHARDS=$(ls "$RUN_DIR"/shard_*.jsonl | wc -l)
[[ "$GPUS" -eq "$SHARDS" ]] || { echo "manifest has $SHARDS shards but machine has $GPUS GPUs" >&2; exit 2; }
for ((g = 0; g < GPUS; g++)); do
    nohup "$PROJECT_ROOT/deploy/run_gpu_worker.sh" "$g" "$RUN" "$MODE" >"$RUN_DIR/worker_gpu$g.log" 2>&1 &
    echo $! >"$RUN_DIR/worker_gpu$g.pid"
    echo "gpu$g pid=$! log=$RUN_DIR/worker_gpu$g.log"
done
