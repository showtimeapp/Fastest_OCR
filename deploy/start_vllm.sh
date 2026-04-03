#!/usr/bin/env bash
# =============================================================================
# Start vLLM Server — Optimized for GLM-OCR on GCP L4 (24GB VRAM)
#
# Optimization flags explained:
#   --dtype float16              FP16 inference (L4 tensor cores)
#   --gpu-memory-utilization 0.92  Use 92% of 24GB VRAM
#   --max-model-len 8192        Max generation length for OCR
#   --max-num-seqs 16           Max concurrent sequences (batch size)
#   --enable-prefix-caching     Cache common prompt prefixes (5-15% speedup)
#   --enable-chunked-prefill    Overlap prefill with decode (better latency)
#   --disable-log-requests      Reduce log spam in production
#   --allowed-local-media-path  Allow local image file references
# =============================================================================
set -euo pipefail

# Activate environment
source /opt/glm-ocr-env/bin/activate 2>/dev/null || true

MODEL="zai-org/GLM-OCR"
PORT="${VLLM_PORT:-8080}"
HOST="${VLLM_HOST:-0.0.0.0}"

echo "============================================="
echo "  Starting vLLM — GLM-OCR"
echo "  Model:  ${MODEL}"
echo "  Listen: ${HOST}:${PORT}"
echo "============================================="

# --- L4 GPU-specific environment variables ---
# Enable TF32 for faster matmuls on Ampere/Ada GPUs
export NVIDIA_TF32_OVERRIDE=1
# Use Flash Attention v2 if available
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# Pre-allocate NCCL buffers
export NCCL_SOCKET_IFNAME=eth0

exec vllm serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype float16 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 8192 \
    --max-num-seqs 16 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --allowed-local-media-path / \
    --trust-remote-code \
    2>&1 | tee ./vllm.log
