#!/usr/bin/env bash
# Run one generation worker on one GPU: keeps a dedicated ComfyUI alive and resumes the shard
# until every record is complete.
#   deploy/run_gpu_worker.sh GPU_INDEX RUN_NAME [smoke|production]
# Env: COMFY_EXTRA_ARGS (default --bf16-unet --highvram; bf16 because Anima overflows to NaN in fp16, and current ComfyUI
# leaves the llm_adapter weights in bf16 when it picks fp16), DATAGEN_ARGS (extra run_worker.py flags).
set -euo pipefail

GPU="${1:?gpu index}"
RUN="${2:?run name}"
MODE="${3:-production}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMFYUI_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
PORT=$((8190 + GPU))
RUN_DIR="$PROJECT_ROOT/runs/$RUN"
COMFY_DIR="$RUN_DIR/comfy_gpu$GPU"
SHARD="$RUN_DIR/shard_${GPU}.jsonl"
OUT="$DATA_ROOT/$RUN"
RESULTS="$OUT/results_gpu${GPU}.jsonl"
mkdir -p "$COMFY_DIR"/{user,input,output,temp} "$OUT"

start_comfy() {
    curl -fsS "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1 && return
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        nohup "$PYTHON_BIN" -u "$COMFYUI_ROOT/main.py" --listen 127.0.0.1 --port "$PORT" \
        --disable-auto-launch --preview-method none ${COMFY_EXTRA_ARGS:---bf16-unet --highvram} \
        --user-directory "$COMFY_DIR/user" --database-url "sqlite:///$COMFY_DIR/user/comfyui.db" \
        --input-directory "$COMFY_DIR/input" --output-directory "$COMFY_DIR/output" --temp-directory "$COMFY_DIR/temp" \
        >"$COMFY_DIR/comfy.log" 2>&1 &
    echo $! >"$COMFY_DIR/comfy.pid"
    for _ in $(seq 1 180); do
        curl -fsS "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1 && return
        sleep 2
    done
    tail -60 "$COMFY_DIR/comfy.log" >&2
    return 1
}

run() {
    "$PYTHON_BIN" -u "$PROJECT_ROOT/datagen/run_worker.py" --shard "$SHARD" --out-dir "$OUT" \
        --results "$RESULTS" --comfy-url "http://127.0.0.1:$PORT" ${DATAGEN_ARGS:-} "$@"
}

start_comfy
if [[ "$MODE" == "smoke" ]]; then
    run --max-batches 2
    exit 0
fi
failures=0
until run; do
    failures=$((failures + 1))
    [[ $failures -ge 3 ]] && { echo "gpu$GPU: three consecutive failures, giving up" >&2; exit 1; }
    kill "$(cat "$COMFY_DIR/comfy.pid")" 2>/dev/null || true
    sleep 5
    start_comfy
done
echo "gpu$GPU: shard complete"
