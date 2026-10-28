#!/usr/bin/env bash
# One-time setup on a Vast instance (PyTorch/CUDA image). Idempotent.
#   PROJECT_ROOT=/workspace/anima-ref-adapter-research bash deploy/bootstrap.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMFYUI_ROOT="${COMFYUI_ROOT:-/workspace/ComfyUI}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
MODELS="$COMFYUI_ROOT/models"

log() { echo "[bootstrap $(date +%H:%M:%S)] $*"; }

log "python: $PYTHON_BIN ($("$PYTHON_BIN" -c 'import torch,sys;print(sys.version.split()[0], torch.__version__, torch.version.cuda)'))"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ ! -f "$COMFYUI_ROOT/main.py" ]]; then
    log "cloning ComfyUI"
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_ROOT"
fi
"$PYTHON_BIN" -m pip install -q -r "$COMFYUI_ROOT/requirements.txt"
"$PYTHON_BIN" -m pip install -q 'huggingface_hub>=0.30' safetensors pillow 'PyWavelets>=1.6'  # PyWavelets: ComfyUI-SPEED has no requirements.txt

# SageAttention 2 (sm_120 build from source if no wheel is available)
if ! "$PYTHON_BIN" -c 'import sageattention' >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install -q 'triton>=3.0' || true
    if ! "$PYTHON_BIN" -m pip install -q 'sageattention==2.2.0' --no-build-isolation; then
        SRC=/workspace/SageAttention
        [[ -d "$SRC/.git" ]] || git clone --depth 1 https://github.com/thu-ml/SageAttention.git "$SRC"
        (cd "$SRC" && CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" \
            MAX_JOBS="$(nproc)" EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 4" "$PYTHON_BIN" setup.py install)
    fi
fi

clone_node() {
    local url="$1" name="$2" target="$COMFYUI_ROOT/custom_nodes/$2"
    [[ -d "$target/.git" ]] || git clone --filter=blob:none "$url" "$target"
    [[ -f "$target/requirements.txt" ]] && "$PYTHON_BIN" -m pip install -q -r "$target/requirements.txt" || true
}
clone_node https://github.com/kijai/ComfyUI-KJNodes.git ComfyUI-KJNodes
clone_node https://github.com/sorryhyun/ComfyUI-Anima-BlockCompile.git ComfyUI-Anima-BlockCompile
clone_node https://github.com/ruwwww/ComfyUI-SPEED.git ComfyUI-SPEED
ln -sfn "$PROJECT_ROOT/comfyui_nodes/anima_ref_datagen_nodes" "$COMFYUI_ROOT/custom_nodes/anima_ref_datagen_nodes"

# Models: Hugging Face circlestone-labs/Anima (base v1.0 sha256 matches the local anima_baseV10).
fetch() {  # repo_path local_dir local_name
    local dst="$MODELS/$2/$3"
    [[ -s "$dst" ]] && return
    mkdir -p "$MODELS/$2"
    hf download circlestone-labs/Anima "$1" --local-dir /workspace/hf_anima >/dev/null
    ln -sfn "/workspace/hf_anima/$1" "$dst"
}
fetch split_files/diffusion_models/anima-base-v1.0.safetensors diffusion_models anima_baseV10.safetensors
fetch split_files/text_encoders/qwen_3_06b_base.safetensors text_encoders qwen_3_06b_base.safetensors
fetch split_files/vae/qwen_image_vae.safetensors vae qwenImageVAE_qwenImageVAE.safetensors
echo "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e  $MODELS/diffusion_models/anima_baseV10.safetensors" | sha256sum -c -

log "bootstrap complete"
