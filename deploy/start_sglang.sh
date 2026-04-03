#!/usr/bin/env bash
# =============================================================================
# Start SGLang Server — Optimized for GLM-OCR on GCP L4 (24GB VRAM)
#
# SGLang advantages over vLLM:
#   - RadixAttention: better prefix caching for repeated prompts
#   - Faster scheduling for vision-language models
#   - Better throughput on small models like GLM-OCR (0.9B)
#
# Try this if vLLM throughput plateaus — swap is easy (same OpenAI API).
# =============================================================================
set -euo pipefail

source /opt/glm-ocr-env/bin/activate 2>/dev/null || true

MODEL="zai-org/GLM-OCR"
PORT="${SGLANG_PORT:-8080}"
HOST="${SGLANG_HOST:-0.0.0.0}"

echo "============================================="
echo "  Starting SGLang — GLM-OCR"
echo "  Model:  ${MODEL}"
echo "  Listen: ${HOST}:${PORT}"
echo "============================================="

export NVIDIA_TF32_OVERRIDE=1

exec python -m sglang.launch_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype float16 \
    --mem-fraction-static 0.88 \
    --max-total-tokens 8192 \
    --context-length 8192 \
    --schedule-policy lpm \
    --trust-remote-code \
    --enable-torch-compile \
    --chunked-prefill-size 4096 \
    2>&1 | tee /var/log/sglang-glmocr.log
