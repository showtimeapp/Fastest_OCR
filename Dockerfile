# =============================================================================
# Dockerfile — GLM-OCR API Server
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        libgl1-mesa-glx libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY src/ src/
COPY config.yaml .

EXPOSE 8000

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "300"]
