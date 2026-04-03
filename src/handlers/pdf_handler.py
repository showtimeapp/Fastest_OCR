"""
PDF Handler — fast async OCR pipeline with table detection.

Pipeline:
  1. Rasterize PDF pages (150 DPI, max 1024px)
  2. Send ALL pages concurrently to vLLM with Text Recognition
  3. Send ALL pages concurrently to vLLM with Table Recognition
  4. Merge results — tables as HTML + Markdown
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from src.config import AppConfig
from src.ocr.engine import OCREngine, OCRResult
from src.ocr.postprocessor import (
    DocumentResult,
    PageResult,
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


def md_table_to_html(md_text: str) -> str:
    """Convert markdown table to HTML table."""
    lines = [l.strip() for l in md_text.strip().split("\n") if l.strip()]
    if not lines:
        return ""

    html_parts = ['<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">']

    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        # Skip separator rows like |---|---|
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("- :") for c in cells):
            continue

        tag = "th" if i == 0 else "td"
        html_parts.append("  <tr>")
        for cell in cells:
            html_parts.append(f"    <{tag}>{cell}</{tag}>")
        html_parts.append("  </tr>")

    html_parts.append("</table>")
    return "\n".join(html_parts)


def has_table_content(text: str) -> bool:
    """Check if OCR text contains a markdown table."""
    lines = text.strip().split("\n")
    pipe_lines = sum(1 for l in lines if "|" in l and l.strip().startswith("|"))
    return pipe_lines >= 2


def extract_tables_from_text(text: str) -> list[dict]:
    """Extract markdown tables from text and convert to HTML."""
    tables = []
    lines = text.strip().split("\n")
    current_table = []
    in_table = False

    for line in lines:
        if line.strip().startswith("|") and "|" in line.strip()[1:]:
            in_table = True
            current_table.append(line)
        else:
            if in_table and current_table:
                md = "\n".join(current_table)
                tables.append({
                    "markdown": md,
                    "html": md_table_to_html(md),
                })
                current_table = []
            in_table = False

    # Last table
    if current_table:
        md = "\n".join(current_table)
        tables.append({
            "markdown": md,
            "html": md_table_to_html(md),
        })

    return tables


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
    logger.info("Processing %d pages from %s", total_pages, filename)

    # --- Send BOTH prompts concurrently ---
    # Text recognition for all pages
    text_task = engine.ocr_batch(
        images=page_images,
        prompts=["Text Recognition:"] * total_pages,
        page_indices=list(range(total_pages)),
        region_indices=[0] * total_pages,
        labels=["text"] * total_pages,
        bboxes=[None] * total_pages,
    )

    # Table recognition for all pages
    table_task = engine.ocr_batch(
        images=page_images,
        prompts=["Table Recognition:"] * total_pages,
        page_indices=list(range(total_pages)),
        region_indices=[0] * total_pages,
        labels=["table"] * total_pages,
        bboxes=[None] * total_pages,
    )

    text_results, table_results = await asyncio.gather(text_task, table_task)

    # --- Assemble page results with table extraction ---
    page_results = []
    for page_idx in range(total_pages):
        text_ocr = text_results[page_idx]
        table_ocr = table_results[page_idx]

        raw_text = text_ocr.text.strip() if text_ocr.text else ""
        table_text = table_ocr.text.strip() if table_ocr.text else ""

        # Extract tables from table recognition
        tables = []
        if table_text and has_table_content(table_text):
            tables = extract_tables_from_text(table_text)
        # Also check text recognition output for tables
        elif raw_text and has_table_content(raw_text):
            tables = extract_tables_from_text(raw_text)

        regions = []
        if raw_text:
            regions.append({
                "index": 0,
                "label": "text",
                "content": raw_text,
            })

        if tables:
            for t_idx, table in enumerate(tables):
                regions.append({
                    "index": t_idx + 1,
                    "label": "table",
                    "content_markdown": table["markdown"],
                    "content_html": table["html"],
                })

        page_results.append(
            PageResult(
                page_number=page_idx + 1,
                raw_text=raw_text,
                regions=regions,
            )
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