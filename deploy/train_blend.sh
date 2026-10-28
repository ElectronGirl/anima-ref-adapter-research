#!/usr/bin/env bash
# Style-blend training once the blend run's generation ends (single GPU).
#   nohup deploy/train_blend.sh BLEND_RUN BASE_PREP TRAIN_NAME [-- train_char_v2.py flags] > runs/train/NAME.chain.log 2>&1 &
# e.g. deploy/train_blend.sh blend1 runs/train/dual_r32_prep char_v3
# Waits for the generation worker, stops its ComfyUI (it holds VRAM), prepares the blend tensors
# once (prepare_blend.py), then trains with configs/train_char_v3.json.
set -euo pipefail
RUN="${1:?blend run}"
BASE="${2:?base prep}"
NAME="${3:?train name}"
shift 3
[[ "${1:-}" == "--" ]] && shift
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMFYUI_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/train_char_v3.json}"
PREP="$PROJECT_ROOT/runs/train/${RUN}_prep"
OUT="$PROJECT_ROOT/runs/train/$NAME"
cd "$PROJECT_ROOT"
mkdir -p "$OUT"

for pidfile in "runs/$RUN"/worker_gpu*.pid; do
    [[ -f "$pidfile" ]] || continue
    while kill -0 "$(cat "$pidfile")" 2>/dev/null; do sleep 60; done
done
for pidfile in "runs/$RUN"/comfy_gpu*/comfy.pid; do
    [[ -f "$pidfile" ]] && kill "$(cat "$pidfile")" 2>/dev/null || true
done
echo "[train_blend $(date +%H:%M:%S)] generation of $RUN finished"
sleep 5

if [[ ! -f "$PREP/index.json" ]]; then
    "$PYTHON_BIN" -u train/prepare_blend.py --base "$BASE" --blend-run "$RUN" --blend-data "$DATA_ROOT/$RUN" \
        --out "$PREP" --comfy-root "$COMFYUI_ROOT" 2>&1 | tee "$OUT/prepare.log"
fi
"$PYTHON_BIN" -u train/train_char_v2.py --prep "$BASE" --blend-prep "$PREP" --config "$CONFIG" --out "$OUT" \
    --comfy-root "$COMFYUI_ROOT" "$@" 2>&1 | tee "$OUT/train.log"
