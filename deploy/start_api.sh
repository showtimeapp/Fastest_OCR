#!/usr/bin/env bash
# =============================================================================
# Start GLM-OCR API Server (FastAPI + Uvicorn)
# =============================================================================
set -euo pipefail

source /opt/glm-ocr-env/bin/activate 2>/dev/null || true

HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"
WORKERS="${API_WORKERS:-1}"

echo "============================================="
echo "  Starting GLM-OCR API"
echo "  Listen: ${HOST}:${PORT}"
echo "  Workers: ${WORKERS}"
echo "============================================="

cd "$(dirname "$0")/.."

exec uvicorn src.server:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --log-level info \
    --timeout-keep-alive 300 \
    2>&1 | tee /var/log/glmocr-api.log
