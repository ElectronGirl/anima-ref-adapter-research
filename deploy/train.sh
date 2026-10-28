#!/usr/bin/env bash
# Train the reference adapter on a finished generation run (single GPU).
#   deploy/train.sh RUN_NAME TRAIN_NAME [train.py flags...]
# Stops the run's ComfyUI processes (they hold VRAM), prepares tensors once, trains, samples.
set -euo pipefail

RUN="${1:?run name}"
NAME="${2:?train name}"
shift 2
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMFYUI_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
PREP="$PROJECT_ROOT/runs/train/${RUN}_prep"
OUT="$PROJECT_ROOT/runs/train/$NAME"
mkdir -p "$OUT"

for pidfile in "$PROJECT_ROOT/runs/$RUN"/comfy_gpu*/comfy.pid; do
    [[ -f "$pidfile" ]] && kill "$(cat "$pidfile")" 2>/dev/null || true
done
sleep 5

if [[ ! -f "$PREP/index.json" ]]; then
    "$PYTHON_BIN" -u "$PROJECT_ROOT/train/prepare.py" --results "$DATA_ROOT/$RUN"/results_gpu*.jsonl \
        --data-dir "$DATA_ROOT/$RUN" --out "$PREP" --comfy-root "$COMFYUI_ROOT" 2>&1 | tee "$OUT/prepare.log"
fi
"$PYTHON_BIN" -u "$PROJECT_ROOT/train/train.py" --prep "$PREP" --out "$OUT" --comfy-root "$COMFYUI_ROOT" "$@" 2>&1 | tee "$OUT/train.log"
