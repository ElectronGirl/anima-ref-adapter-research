#!/usr/bin/env bash
# Wait for the generation worker of RUN to exit, then train TRAIN_NAME on its output.
# If training fails (e.g. out of memory), retry once with full gradient checkpointing.
#   nohup deploy/after_pilot.sh RUN TRAIN_NAME > runs/train/TRAIN_NAME.chain.log 2>&1 &
set -uo pipefail
RUN="${1:?run name}"
NAME="${2:?train name}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
for pidfile in "$PROJECT_ROOT/runs/$RUN"/worker_gpu*.pid; do
    while kill -0 "$(cat "$pidfile")" 2>/dev/null; do sleep 60; done
done
echo "[chain $(date +%H:%M:%S)] generation finished; training $NAME"
"$PROJECT_ROOT/deploy/train.sh" "$RUN" "$NAME" && exit 0
echo "[chain $(date +%H:%M:%S)] training failed; retrying with all blocks checkpointed"
rm -f "$PROJECT_ROOT/runs/train/$NAME"/train_log.jsonl
"$PROJECT_ROOT/deploy/train.sh" "$RUN" "$NAME" --checkpoint-blocks 28
