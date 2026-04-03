#!/usr/bin/env bash
# =============================================================================
# GCP L4 VM Setup Script
# Run once on a fresh VM to install all dependencies
# Tested on: Ubuntu 24.04 LTS (Python 3.12) / Ubuntu 22.04 LTS (Python 3.10)
# =============================================================================
set -euo pipefail

echo "============================================="
echo "  GLM-OCR Production Setup — GCP L4 VM"
echo "============================================="

# --- Detect Python version ---
PYTHON_BIN=""
for py in python3.12 python3.11 python3.10 python3; do
    if command -v "$py" &> /dev/null; then
        PYTHON_BIN="$py"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: No Python 3.10+ found. Install Python first."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
echo "Detected Python: $PYTHON_BIN ($PYTHON_VERSION)"

# --- System packages ---
echo ""
echo "[1/7] Installing system packages..."
sudo apt-get update -qq

# libgl1-mesa-glx was renamed to libgl1 in Ubuntu 24.04
GL_PKG="libgl1"

sudo apt-get install -y -qq \
    python3-venv python3-dev python3-pip \
    git wget curl \
    "$GL_PKG" libglib2.0-0 \
    poppler-utils

echo "  System packages installed ✓"

# --- CUDA / NVIDIA check ---
echo ""
echo "[2/7] Checking NVIDIA drivers..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "  NVIDIA drivers found ✓"
else
    echo "  WARNING: nvidia-smi not found."
    echo "  If GPU is attached, install drivers:"
    echo "    sudo apt-get install -y nvidia-driver-535"
    echo "    sudo reboot"
    echo "  Then re-run this script."
    echo ""
    echo "  Continuing anyway (will fail at vLLM step if no GPU)..."
fi

# --- Python virtual environment ---
echo ""
echo "[3/7] Setting up Python virtual environment..."
VENV_PATH="/opt/glm-ocr-env"

if [ -d "$VENV_PATH" ]; then
    echo "  Virtual env already exists at $VENV_PATH, reusing..."
else
    sudo $PYTHON_BIN -m venv "$VENV_PATH"
    sudo chown -R "$(whoami):$(whoami)" "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
pip install --upgrade pip setuptools wheel -q
echo "  Virtual env ready at $VENV_PATH ✓"
echo "  Using: $(python --version) at $(which python)"

# --- Install transformers from source (required for GLM-OCR) ---
echo ""
echo "[4/7] Installing transformers (from source — required for GLM-OCR)..."
pip install git+https://github.com/huggingface/transformers.git -q
echo "  transformers installed ✓"

# --- Install vLLM ---
echo ""
echo "[5/7] Installing vLLM..."
pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly -q
echo "  vLLM installed ✓"

# --- Install project dependencies ---
echo ""
echo "[6/7] Installing project dependencies..."

# Auto-detect project directory from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
echo "  Project directory: $PROJECT_DIR"

cd "$PROJECT_DIR"
pip install -r requirements.txt -q
echo "  Dependencies installed ✓"

# --- Download model (pre-cache to avoid first-run delay) ---
echo ""
echo "[7/7] Pre-downloading GLM-OCR model..."
python -c "
from transformers import AutoProcessor, AutoModelForImageTextToText
print('  Downloading processor...')
AutoProcessor.from_pretrained('zai-org/GLM-OCR')
print('  Downloading model...')
AutoModelForImageTextToText.from_pretrained('zai-org/GLM-OCR')
print('  Model cached successfully ✓')
"

echo ""
echo "============================================="
echo "  Setup complete!"
echo "============================================="
echo ""
echo "  Python:  $(python --version)"
echo "  venv:    $VENV_PATH"
echo "  Project: $PROJECT_DIR"
echo ""
echo "  To activate the environment:"
echo "    source $VENV_PATH/bin/activate"
echo ""
echo "  Next steps:"
echo "    1. Start vLLM:  bash deploy/start_vllm.sh"
echo "    2. Start API:   bash deploy/start_api.sh"
echo "    3. Test:         curl http://localhost:8000/ocr/health"
echo ""
