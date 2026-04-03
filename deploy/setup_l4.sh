#!/usr/bin/env bash
# =============================================================================
# GCP L4 VM Setup Script
# Run once on a fresh VM to install all dependencies
# =============================================================================
set -euo pipefail

echo "============================================="
echo "  GLM-OCR Production Setup — GCP L4 VM"
echo "============================================="

# --- System packages ---
echo "[1/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip git wget curl \
    libgl1-mesa-glx libglib2.0-0 \
    nginx

# --- CUDA check ---
echo "[2/7] Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Install NVIDIA drivers first."
    echo "  sudo apt-get install -y nvidia-driver-535"
    echo "  Reboot, then re-run this script."
fi
nvidia-smi || true

# --- Python virtual environment ---
echo "[3/7] Setting up Python environment..."
python3.11 -m venv /opt/glm-ocr-env
source /opt/glm-ocr-env/bin/activate

pip install --upgrade pip setuptools wheel

# --- Install transformers from source (required for GLM-OCR) ---
echo "[4/7] Installing transformers (source)..."
pip install git+https://github.com/huggingface/transformers.git

# --- Install vLLM (nightly for vision model support) ---
echo "[5/7] Installing vLLM (nightly)..."
pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly

# --- Install project dependencies ---
echo "[6/7] Installing project dependencies..."
cd /opt/glm-ocr-service
pip install -r requirements.txt

# --- Download model (pre-cache to avoid first-run delay) ---
echo "[7/7] Pre-downloading GLM-OCR model..."
python -c "
from transformers import AutoProcessor, AutoModelForImageTextToText
print('Downloading processor...')
AutoProcessor.from_pretrained('zai-org/GLM-OCR')
print('Downloading model...')
AutoModelForImageTextToText.from_pretrained('zai-org/GLM-OCR')
print('Model cached successfully.')
"

echo ""
echo "============================================="
echo "  Setup complete!"
echo "============================================="
echo ""
echo "Next steps:"
echo "  1. Start vLLM:  bash deploy/start_vllm.sh"
echo "  2. Start API:   bash deploy/start_api.sh"
echo "  3. Test:        curl http://localhost:8000/ocr/health"
echo ""
