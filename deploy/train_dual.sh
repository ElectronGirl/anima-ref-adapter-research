#!/usr/bin/env bash
# Train the two-reference adapter on finished generation runs (single GPU).
#   deploy/train_dual.sh TRAIN_NAME SOURCE_RUN... [-- train_dual.py flags...]
# e.g. deploy/train_dual.sh dual_r32 pilot grid1 -- --init-lora runs/train/pilot_r32/lora_step02000.safetensors
# Waits for the runs' generation workers, stops their ComfyUI processes (they hold VRAM),
# prepares tensors once, trains, samples.
set -euo pipefail

NAME="${1:?train name}"
shift
RUNS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do RUNS+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMFYUI_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
PREP="$PROJECT_ROOT/runs/train/${NAME}_prep"
OUT="$PROJECT_ROOT/runs/train/$NAME"
mkdir -p "$OUT"

SOURCES=()
for run in "${RUNS[@]}"; do
    for pidfile in "$PROJECT_ROOT/runs/$run"/worker_gpu*.pid; do
        [[ -f "$pidfile" ]] || continue
        while kill -0 "$(cat "$pidfile")" 2>/dev/null; do sleep 60; done
    done
    for pidfile in "$PROJECT_ROOT/runs/$run"/comfy_gpu*/comfy.pid; do
        [[ -f "$pidfile" ]] && kill "$(cat "$pidfile")" 2>/dev/null || true
    done
    SOURCES+=(--source "$run=$DATA_ROOT/$run")
done
echo "[train_dual $(date +%H:%M:%S)] generation finished; preparing ${RUNS[*]}"
sleep 5

if [[ ! -f "$PREP/index.json" ]]; then
    "$PYTHON_BIN" -u "$PROJECT_ROOT/train/prepare_dual.py" "${SOURCES[@]}" --out "$PREP" \
        --comfy-root "$COMFYUI_ROOT" 2>&1 | tee "$OUT/prepare.log"
fi
"$PYTHON_BIN" -u "$PROJECT_ROOT/train/train_dual.py" --prep "$PREP" --out "$OUT" --comfy-root "$COMFYUI_ROOT" "$@" \
    2>&1 | tee "$OUT/train.log"
