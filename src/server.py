"""
FastAPI Server — production HTTP API for GLM-OCR.

Endpoints:
  POST /ocr/process     — upload file(s), get OCR results
  POST /ocr/process-url — process file from URL
  GET  /ocr/health      — health check
  GET  /ocr/status      — backend + config status
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse

from src.config import AppConfig, load_config
from src.ocr.engine import OCREngine
from src.ocr.layout import LayoutDetector
from src.ocr.postprocessor import to_json, to_markdown
from src.router import detect_file_type, process_file, SUPPORTED_EXTENSIONS

import json
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# --- Global state ---
_config: Optional[AppConfig] = None
_engine: Optional[OCREngine] = None
_layout_detector: Optional[LayoutDetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global _config, _engine, _layout_detector

    _config = load_config()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, _config.logging.level.upper(), logging.INFO),
        format=_config.logging.format,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Init layout detector
    _layout_detector = LayoutDetector()
    if _config.pdf.enable_layout:
        _layout_detector.load()

    # Init OCR engine
    _engine = OCREngine(config=_config)
    await _engine.start()

    healthy = await _engine.health_check()
    if healthy:
        logger.info("Backend health check passed ✓")
    else:
        logger.warning(
            "Backend not reachable at %s:%d — OCR requests will fail until backend is up",
            _config.backend.host,
            _config.backend.port,
        )

    logger.info("GLM-OCR API server started on %s:%d", _config.server.host, _config.server.port)

    yield

    # Shutdown
    await _engine.stop()
    logger.info("GLM-OCR API server stopped")


app = FastAPI(
    title="GLM-OCR Production API",
    version="1.0.0",
    description="High-performance OCR service powered by GLM-OCR + vLLM/SGLang",
    lifespan=lifespan,
)


@app.get("/ocr/health")
async def health_check():
    """Health check endpoint."""
    backend_ok = await _engine.health_check() if _engine else False
    status = "healthy" if backend_ok else "degraded"
    return {
        "status": status,
        "backend_reachable": backend_ok,
        "backend_type": _config.backend.type if _config else None,
    }


@app.get("/ocr/status")
async def status():
    """Detailed status including config info."""
    return {
        "backend": {
            "type": _config.backend.type,
            "host": _config.backend.host,
            "port": _config.backend.port,
            "max_concurrent": _config.backend.max_concurrent_requests,
        },
        "model": {
            "name": _config.model.name,
            "dtype": _config.model.dtype,
        },
        "pdf": {
            "layout_enabled": _config.pdf.enable_layout,
            "layout_available": _layout_detector.available if _layout_detector else False,
            "render_dpi": _config.pdf.render_dpi,
            "region_batch_size": _config.pdf.region_batch_size,
        },
        "output_format": _config.output.format,
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    }


@app.post("/ocr/process")
async def process_upload(
    file: UploadFile = File(...),
    output_format: Optional[str] = Query(
        default=None,
        description="Override output format: json, markdown, or both",
    ),
):
    """
    Process an uploaded file.

    Accepts: .pdf, .doc, .docx, .txt
    Returns: JSON with OCR results (and optionally markdown)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    # Check file size
    max_bytes = _config.server.max_upload_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {_config.server.max_upload_size_mb}MB",
        )

    # Save to temp file
    request_id = uuid.uuid4().hex[:12]
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"glmocr_{request_id}_"))

    try:
        tmp_file = tmp_dir / file.filename
        tmp_file.write_bytes(content)

        # Use override format or config default
        original_format = _config.output.format
        if output_format:
            _config.output.format = output_format

        # Process
        doc_result, saved_files = await process_file(
            file_path=tmp_file,
            engine=_engine,
            config=_config,
            layout_detector=_layout_detector,
            output_dir=tmp_dir / "output",
        )

        # Restore config
        _config.output.format = original_format

        # Build response
        response = {
            "request_id": request_id,
            "result": to_json(doc_result),
        }

        # Include markdown if requested
        if output_format in ("markdown", "both") or _config.output.format in ("markdown", "both"):
            response["markdown"] = to_markdown(doc_result)

        # If download requested, return files
        if output_format == "markdown":
            md_content = to_markdown(doc_result)
            return Response(
                content=md_content,
                media_type="text/markdown",
                headers={"Content-Disposition": f"attachment; filename={Path(file.filename).stem}.md"}
            )
        
        if output_format == "json":
            json_content = to_json(doc_result)
            return Response(
                content=json.dumps(json_content, ensure_ascii=False, indent=2),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename={Path(file.filename).stem}.json"}
            )

        # Default: return both in JSON response
        return JSONResponse(content=response)

    except Exception as exc:
        logger.exception("Processing failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        # Cleanup temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/ocr/process-batch")
async def process_batch(
    files: list[UploadFile] = File(...),
    output_format: Optional[str] = Query(default=None),
):
    """
    Process multiple uploaded files.
    Each file is processed sequentially; OCR within each PDF is parallel.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    request_id = uuid.uuid4().hex[:12]
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"glmocr_batch_{request_id}_"))

    try:
        results = []
        for file in files:
            if not file.filename:
                continue

            suffix = Path(file.filename).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                results.append({
                    "filename": file.filename,
                    "error": f"Unsupported file type: {suffix}",
                })
                continue

            content = await file.read()
            tmp_file = tmp_dir / file.filename
            tmp_file.write_bytes(content)

            try:
                original_format = _config.output.format
                if output_format:
                    _config.output.format = output_format

                doc_result, _ = await process_file(
                    file_path=tmp_file,
                    engine=_engine,
                    config=_config,
                    layout_detector=_layout_detector,
                    output_dir=tmp_dir / "output",
                )

                _config.output.format = original_format

                entry = {"filename": file.filename, "result": to_json(doc_result)}
                if output_format in ("markdown", "both"):
                    entry["markdown"] = to_markdown(doc_result)
                results.append(entry)

            except Exception as exc:
                logger.exception("Failed: %s", file.filename)
                results.append({"filename": file.filename, "error": str(exc)})

        return JSONResponse(content={"request_id": request_id, "results": results})

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def start_server(config_path: str = None):
    """Start the server programmatically."""
    import uvicorn

    config = load_config(config_path)
    uvicorn.run(
        "src.server:app",
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        log_level=config.logging.level.lower(),
    )


if __name__ == "__main__":
    start_server()
