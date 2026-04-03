# GLM-OCR Production Service

High-performance document OCR service powered by **GLM-OCR (0.9B)** with **vLLM/SGLang** inference backends, optimized for **GCP L4 GPU** (24GB VRAM).

## Features

- **Smart file routing**: PDF → OCR pipeline, DOC/TXT → direct extraction (no GPU)
- **Layout-aware OCR**: PP-DocLayout-V3 splits pages into regions for parallel processing
- **Dual output**: Markdown (`.md`) + structured JSON (`.json`) for every document
- **Production optimized**: FP16, prefix caching, chunked prefill, Flash Attention
- **Async pipeline**: CPU rasterization overlaps with GPU inference — zero idle time
- **FastAPI server**: Upload files via REST API with batch support
- **Benchmarking**: Built-in tools to compare backends and configurations

## Architecture

```
Input File
    │
    ├── .txt/.md ──→ Direct read ──→ Output
    ├── .doc/.docx ─→ python-docx ──→ Output
    └── .pdf ──────→ OCR Pipeline:
                      │
                      ├─ 1. PyMuPDF rasterize (8 CPU threads)
                      ├─ 2. PP-DocLayout-V3 (region detection)
                      ├─ 3. Crop + preprocess regions
                      ├─ 4. GLM-OCR via vLLM/SGLang (async batch)
                      └─ 5. Assemble → Markdown + JSON
```

## Quick Start

### 1. Setup (GCP L4 VM)

```bash
# Clone project
git clone <your-repo> /opt/glm-ocr-service
cd /opt/glm-ocr-service

# Run setup script (installs everything)
bash deploy/setup_l4.sh
```

### 2. Start vLLM Backend

```bash
bash deploy/start_vllm.sh
# Wait ~60s for model to load, then check:
curl http://localhost:8080/health
```

### 3. Process Documents

**CLI:**
```bash
# Single file
python -m src.main document.pdf

# Multiple files
python -m src.main file1.pdf file2.docx notes.txt

# Directory
python -m src.main ./documents/ --output ./results/

# Debug mode (shows timing for each stage)
python -m src.main document.pdf --log-level DEBUG
```

**API Server:**
```bash
# Start server
bash deploy/start_api.sh

# Upload a file
curl -X POST http://localhost:8000/ocr/process \
  -F "file=@document.pdf" | python -m json.tool

# Batch upload
curl -X POST http://localhost:8000/ocr/process-batch \
  -F "files=@file1.pdf" \
  -F "files=@file2.pdf"
```

**Docker:**
```bash
docker compose up -d
# vLLM on :8080, API on :8000
```

## Configuration

Edit `config.yaml`:

```yaml
backend:
  type: vllm              # or sglang
  host: localhost
  port: 8080
  max_concurrent_requests: 16

model:
  dtype: float16           # FP16 for L4 tensor cores

pdf:
  render_dpi: 300
  rasterize_workers: 8     # match vCPU count
  enable_layout: true      # PP-DocLayout-V3 for 2-3x speed
  region_batch_size: 8

output:
  format: both             # json, markdown, or both
  output_dir: ./output
```

## Benchmarking

```bash
# Basic benchmark (3 iterations)
python -m benchmark.benchmark test.pdf

# Compare layout ON vs OFF
python -m benchmark.benchmark test.pdf --compare-layout

# Compare vLLM vs SGLang
# (run vLLM on :8080, SGLang on :8081 first)
python -m benchmark.compare_backends test.pdf

# Test different concurrency levels
python -m benchmark.benchmark test.pdf --batch-sizes 1,4,8,16
```

## Optimization Stack

| Layer | Method | Speedup | Where |
|-------|--------|---------|-------|
| 1 | FP16 inference | +30-40% | vLLM `--dtype float16` |
| 2 | Layout detection + parallel OCR | +50-100% | `enable_layout: true` |
| 3 | Prefix caching | +5-15% | vLLM `--enable-prefix-caching` |
| 4 | Chunked prefill | +5-10% | vLLM `--enable-chunked-prefill` |
| 5 | Flash Attention 2 | +15-25% | Auto (vLLM default) |
| 6 | CPU-GPU pipeline overlap | +10-20% | Async rasterize + OCR |
| 7 | torch.compile (SGLang) | +10-20% | SGLang `--enable-torch-compile` |
| 8 | TF32 matmul | +5-10% | `NVIDIA_TF32_OVERRIDE=1` |

## Output Format

**Markdown** (`result.md`):
```markdown
# document.pdf
> Processed 10 pages in 5.32s (1.88 pages/sec)

---
<!-- Page 1 -->
# Document Title
Body text here...

| Table | Content |
|-------|---------|
| A     | B       |
```

**JSON** (`result.json`):
```json
{
  "metadata": {
    "filename": "document.pdf",
    "total_pages": 10,
    "processing_time_sec": 5.32,
    "pages_per_second": 1.88
  },
  "pages": [
    {
      "page_number": 1,
      "raw_text": "...",
      "regions": [
        { "index": 0, "label": "title", "content": "..." },
        { "index": 1, "label": "text", "content": "..." }
      ]
    }
  ]
}
```

## Project Structure

```
glm-ocr-service/
├── config.yaml              # All tunable parameters
├── docker-compose.yml       # Full stack deployment
├── Dockerfile               # API server container
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── main.py              # CLI entry point
│   ├── server.py            # FastAPI HTTP server
│   ├── config.py            # Pydantic config loader
│   ├── router.py            # File type routing
│   ├── handlers/
│   │   ├── pdf_handler.py   # PDF OCR pipeline
│   │   ├── doc_handler.py   # DOC/DOCX extraction
│   │   └── text_handler.py  # TXT passthrough
│   └── ocr/
│       ├── engine.py        # vLLM/SGLang async client
│       ├── layout.py        # PP-DocLayout-V3 detector
│       ├── preprocessor.py  # Image resize + normalize
│       └── postprocessor.py # Results → Markdown + JSON
├── deploy/
│   ├── setup_l4.sh          # GCP L4 VM setup
│   ├── start_vllm.sh        # vLLM with optimizations
│   ├── start_sglang.sh      # SGLang with optimizations
│   └── start_api.sh         # API server
├── benchmark/
│   ├── benchmark.py         # Throughput benchmarking
│   └── compare_backends.py  # vLLM vs SGLang comparison
└── tests/
    └── test_pipeline.py     # Unit tests
```

## vLLM vs SGLang

Both are officially supported by GLM-OCR. Start with vLLM (more stable), then benchmark SGLang:

| | vLLM | SGLang |
|--|------|--------|
| Prefix caching | Basic | Superior (RadixAttention) |
| torch.compile | Via backend | First-class flag |
| Stability | More mature | Faster iteration |
| Vision models | Good | Good |

Switch backends by changing one line in `config.yaml`:
```yaml
backend:
  type: sglang  # was: vllm
  port: 8081    # SGLang port
```

## License

MIT (GLM-OCR model) + Apache 2.0 (PP-DocLayout-V3 + this codebase)
