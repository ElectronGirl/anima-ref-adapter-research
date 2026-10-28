#!/usr/bin/env bash
# Single-purpose adapter training (train/train_ref.py) once an extra hires run's generation ends.
#   nohup deploy/train_ref.sh TASK HIRES_RUN BASE_PREP CONFIG TRAIN_NAME [-- train_ref.py flags] > runs/train/NAME.chain.log 2>&1 &
# e.g. deploy/train_ref.sh identity hires1 runs/train/dual_r32_prep configs/train_char_only.json char_only
set -euo pipefail
TASK="${1:?identity|copy}"
RUN="${2:?hires run}"
BASE="${3:?base prep}"
CONFIG="${4:?config}"
NAME="${5:?train name}"
shift 5
[[ "${1:-}" == "--" ]] && shift
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMFYUI_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
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
echo "[train_ref $(date +%H:%M:%S)] generation of $RUN finished"
sleep 5
"$PYTHON_BIN" -u train/train_ref.py --task "$TASK" --prep "$BASE" --hires-run "$RUN" --hires-data "$DATA_ROOT/$RUN" \
    --config "$CONFIG" --out "$OUT" --comfy-root "$COMFYUI_ROOT" "$@" 2>&1 | tee "$OUT/train.log"
