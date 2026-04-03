"""
PDF Handler — the core OCR pipeline.

Pipeline:
  1. Rasterize PDF pages to images (parallel CPU threads)
  2. Detect layout regions per page (PP-DocLayout-V3)
  3. Crop regions and send to OCR engine (async batched GPU)
  4. Assemble results into DocumentResult

This is where PDF → Markdown + JSON happens.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

from src.config import AppConfig
from src.ocr.engine import OCREngine
from src.ocr.layout import (
    LayoutDetector,
    crop_region,
    filter_ocr_regions,
    get_prompt_for_label,
)
from src.ocr.preprocessor import preprocess_image
from src.ocr.postprocessor import (
    DocumentResult,
    assemble_page_results,
    build_document_result,
)

logger = logging.getLogger(__name__)


def rasterize_page(
    pdf_path: str,
    page_index: int,
    dpi: int = 300,
) -> Image.Image:
    """
    Rasterize a single PDF page to a PIL Image.

    Uses PyMuPDF (fitz) — the fastest Python PDF renderer.
    Each call opens/closes the PDF to be thread-safe.
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return img
    finally:
        doc.close()


def rasterize_all_pages(
    pdf_path: str,
    dpi: int = 300,
    max_workers: int = 8,
) -> list[Image.Image]:
    """
    Rasterize all PDF pages in parallel using a thread pool.

    Returns list of PIL Images in page order.
    """
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()

    logger.info("Rasterizing %d pages at %d DPI with %d workers", num_pages, dpi, max_workers)
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(rasterize_page, pdf_path, i, dpi)
            for i in range(num_pages)
        ]
        images = [f.result() for f in futures]

    elapsed = time.perf_counter() - t0
    logger.info(
        "Rasterization complete: %d pages in %.2fs (%.1f pages/sec)",
        num_pages,
        elapsed,
        num_pages / elapsed if elapsed > 0 else 0,
    )
    return images


async def process_pdf(
    pdf_path: str | Path,
    engine: OCREngine,
    config: AppConfig,
    layout_detector: Optional[LayoutDetector] = None,
) -> DocumentResult:
    """
    Full PDF processing pipeline.

    1. Rasterize all pages (CPU, parallel threads)
    2. For each page:
       a. Run layout detection → get regions
       b. Crop + preprocess each region
       c. Send to OCR engine (async, batched)
    3. Assemble results

    Returns a DocumentResult ready for Markdown/JSON export.
    """
    pdf_path = Path(pdf_path)
    filename = pdf_path.name
    t_start = time.perf_counter()

    # --- Step 1: Rasterize ---
    loop = asyncio.get_event_loop()
    page_images = await loop.run_in_executor(
        None,
        rasterize_all_pages,
        str(pdf_path),
        config.pdf.render_dpi,
        config.pdf.rasterize_workers,
    )

    total_pages = len(page_images)
    logger.info("Processing %d pages from %s", total_pages, filename)

    # --- Step 2 & 3: Layout Detection + OCR ---
    use_layout = config.pdf.enable_layout and layout_detector and layout_detector.available

    # Collect all OCR tasks across all pages
    all_images = []
    all_prompts = []
    all_page_indices = []
    all_region_indices = []
    all_labels = []
    all_bboxes = []

    for page_idx, page_img in enumerate(page_images):
        if use_layout:
            # Detect regions
            regions = layout_detector.detect(page_img)
            ocr_regions = filter_ocr_regions(regions)

            if not ocr_regions:
                # No OCR-able regions detected — use full page
                preprocessed = preprocess_image(
                    page_img,
                    config.pdf.min_pixels,
                    config.pdf.max_pixels,
                )
                all_images.append(preprocessed)
                all_prompts.append("Text Recognition:")
                all_page_indices.append(page_idx)
                all_region_indices.append(0)
                all_labels.append("full_page")
                all_bboxes.append(None)
                continue

            for reg_idx, region in enumerate(ocr_regions):
                # Crop the region from the page
                cropped = crop_region(page_img, region.bbox)
                preprocessed = preprocess_image(
                    cropped,
                    config.pdf.min_pixels,
                    config.pdf.max_pixels,
                )

                prompt = get_prompt_for_label(region.label)

                all_images.append(preprocessed)
                all_prompts.append(prompt)
                all_page_indices.append(page_idx)
                all_region_indices.append(reg_idx)
                all_labels.append(region.label)
                all_bboxes.append(region.bbox)
        else:
            # No layout detection — send full page
            preprocessed = preprocess_image(
                page_img,
                config.pdf.min_pixels,
                config.pdf.max_pixels,
            )
            all_images.append(preprocessed)
            all_prompts.append("Text Recognition:")
            all_page_indices.append(page_idx)
            all_region_indices.append(0)
            all_labels.append("full_page")
            all_bboxes.append(None)

    logger.info(
        "Dispatching %d OCR tasks (%d pages, layout=%s)",
        len(all_images),
        total_pages,
        "ON" if use_layout else "OFF",
    )

    # --- Batch OCR (async, concurrency-controlled) ---
    ocr_results = await engine.ocr_batch(
        images=all_images,
        prompts=all_prompts,
        page_indices=all_page_indices,
        region_indices=all_region_indices,
        labels=all_labels,
        bboxes=all_bboxes,
    )

    # --- Step 4: Assemble ---
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
        "PDF processing complete: %d pages in %.2fs (%.2f pages/sec)",
        total_pages,
        processing_time,
        doc_result.pages_per_second,
    )

    return doc_result


def get_pdf_page_count(pdf_path: str | Path) -> int:
    """Get the number of pages in a PDF without rasterizing."""
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count
