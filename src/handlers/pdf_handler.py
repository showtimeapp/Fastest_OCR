"""
PDF Handler — fast async OCR pipeline.

Pipeline:
  1. Rasterize PDF pages (parallel CPU threads, 150 DPI)
  2. Resize to max 1024px longest side
  3. Send ALL pages concurrently to vLLM (async)
  4. Assemble results
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from src.config import AppConfig
from src.ocr.engine import OCREngine
from src.ocr.postprocessor import (
    DocumentResult,
    assemble_page_results,
    build_document_result,
)

logger = logging.getLogger(__name__)

MAX_SIDE = 1024


def rasterize_all_pages(
    pdf_path: str,
    dpi: int = 150,
) -> list[Image.Image]:
    doc = fitz.open(pdf_path)
    images = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        w, h = img.size
        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        images.append(img)
    doc.close()
    return images


async def process_pdf(
    pdf_path: str | Path,
    engine: OCREngine,
    config: AppConfig,
    layout_detector=None,
) -> DocumentResult:
    pdf_path = Path(pdf_path)
    filename = pdf_path.name
    t_start = time.perf_counter()

    loop = asyncio.get_event_loop()
    page_images = await loop.run_in_executor(
        None,
        rasterize_all_pages,
        str(pdf_path),
        config.pdf.render_dpi,
    )

    total_pages = len(page_images)
    logger.info("Processing %d pages from %s (size=%s)", total_pages, filename, page_images[0].size if page_images else "?")

    ocr_results = await engine.ocr_batch(
        images=page_images,
        prompts=["Text Recognition:"] * total_pages,
        page_indices=list(range(total_pages)),
        region_indices=[0] * total_pages,
        labels=["full_page"] * total_pages,
        bboxes=[None] * total_pages,
    )

    page_results = assemble_page_results(
        ocr_results,
        total_pages=total_pages,
        include_bboxes=config.output.include_bboxes,
    )

    t_end = time.perf_counter()
    processing_time = t_end - t_start

    doc_result = build_document_result(
        filename=filename,
        page_results=page_results,
        processing_time=processing_time,
        file_type="pdf",
    )

    logger.info(
        "PDF complete: %d pages in %.2fs (%.2f pages/sec)",
        total_pages,
        processing_time,
        doc_result.pages_per_second,
    )

    return doc_result


def get_pdf_page_count(pdf_path: str | Path) -> int:
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count
