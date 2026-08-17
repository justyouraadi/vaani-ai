#!/usr/bin/env bash
# VaaniAI RunPod deployer for a SINGLE A40 (48 GB) pod.
# Template: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
# Everything lives under /workspace (the persistent volume) so pod restarts
# keep models/cache/logs. After a restart: re-run the same command
# (it is idempotent — only starts services that are not already up).
#
#   bash setup_runpod.sh all   # vLLM brain + edge server, one pod (default)
#
# Env overrides:
#   VAANI_MODEL                LLM served by vLLM (default Qwen/Qwen2.5-14B-Instruct)
#   VAANI_VLLM_GPU_UTIL        vLLM share of the 48 GB (default 0.72; edge needs ~6 GB)
#   VAANI_LLM_BASE_URL         brain URL for the edge server (default http://localhost:8000/v1)
set -euo pipefail

MODE="${1:-all}"
MODEL="${VAANI_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
BRAIN_URL="${VAANI_LLM_BASE_URL:-http://localhost:8000/v1}"
VLLM_UTIL="${VAANI_VLLM_GPU_UTIL:-0.72}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${REPO}"
mkdir -p "$WORK"/{logs,run}
export HF_HOME="${HF_HOME:-$WORK/hf_home}"
export TORCH_HOME="${TORCH_HOME:-$WORK/torch_home}"
mkdir -p "$HF_HOME" "$TORCH_HOME"

log() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }

if [ ! -x "$WORK/.venv/bin/python" ]; then
  log "creating venv (python3.11)"
  python3.11 -m venv "$WORK/.venv"
fi
# shellcheck disable=SC1091
. "$WORK/.venv/bin/activate"
pip install -U pip wheel

case "$MODE" in
  all|brain)
    if ! python -c "import vllm" 2>/dev/null; then
      log "installing vllm 0.7.3 (pulls torch 2.5.1 + torchaudio, CUDA 12.4; ~5 min)"
      pip install "vllm==0.7.3"
    fi
    if ! curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
      log "starting vLLM: $MODEL (logs: tail -f $WORK/logs/vaani-brain.log)"
      nohup vllm serve "$MODEL" \
        --host 0.0.0.0 --port 8000 \
        --gpu-memory-utilization "$VLLM_UTIL" \
        --max-model-len 4096 \
        --max-num-seqs 4 \
        --enable-prefix-caching \
        >> "$WORK/logs/vaani-brain.log" 2>&1 &
      echo $! > "$WORK/run/vaani-brain.pid"
    else
      log "vLLM already running on :8000"
    fi
    ;;
esac

case "$MODE" in
  all|edge)
    if ! python -c "import torch" 2>/dev/null; then
      log "installing torch 2.5.1 + torchaudio (matches vllm 0.7.3)"
      pip install "torch==2.5.1" "torchaudio==2.5.1" --index-url https://download.pytorch.org/whl/cu124
    fi
    log "installing faster-whisper, silero-vad, coqui-tts"
    pip install -r requirements.txt
    pip install coqui-tts
    if [ ! -f models/whisper/model.bin ]; then
      log "downloading STT/TTS weights (~5 GB into $HF_HOME)"
      python scripts/download_models.py
    fi
    export VAANI_LLM_BASE_URL="$BRAIN_URL"
    export VAANI_LLM_MODEL="$MODEL"
    if ! curl -sf http://localhost:8765/health >/dev/null 2>&1; then
      log "starting edge server on :8765 (logs: tail -f $WORK/logs/vaani-edge.log)"
      nohup uvicorn vaani.main:app --host 0.0.0.0 --port 8765 \
        >> "$WORK/logs/vaani-edge.log" 2>&1 &
      echo $! > "$WORK/run/vaani-edge.pid"
    else
      log "edge already running on :8765"
    fi
    ;;
esac

log "done. Check:"
echo "  curl -s localhost:8000/v1/models   # vLLM (brain)"
echo "  curl -s localhost:8765/health       # edge"
echo "  tail -f $WORK/logs/vaani-brain.log"
echo "  tail -f $WORK/logs/vaani-edge.log"
echo "  nvidia-smi                          # VRAM split check"
echo
echo "If vLLM OOMs or edge calls fail: set VAANI_MODEL=Qwen/Qwen2.5-7B-Instruct"
echo "Please confirm models/vaani.wav exists in $REPO/models/"
echo "After a pod restart, re-run: bash $0 all"