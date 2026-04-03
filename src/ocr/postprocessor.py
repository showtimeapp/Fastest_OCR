"""
Post-Processor — assembles OCR results into Markdown and JSON output.

Handles:
  - Merging region-level OCR results back into page order
  - Markdown document assembly with page breaks
  - Structured JSON output with metadata
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.ocr.engine import OCRResult

logger = logging.getLogger(__name__)


@dataclass
class PageResult:
    """Processed result for a single page."""
    page_number: int
    raw_text: str
    regions: list[dict] = field(default_factory=list)


@dataclass
class DocumentResult:
    """Complete processed document."""
    filename: str
    total_pages: int
    pages: list[PageResult]
    processing_time_sec: float = 0.0
    pages_per_second: float = 0.0
    file_type: str = "pdf"


def assemble_page_results(
    ocr_results: list[OCRResult],
    total_pages: int,
    include_bboxes: bool = True,
) -> list[PageResult]:
    """
    Group OCR results by page and assemble into PageResult objects.
    Results are sorted by page_index, then region_index.
    """
    # Group by page
    pages_map: dict[int, list[OCRResult]] = {}
    for r in ocr_results:
        pages_map.setdefault(r.page_index, []).append(r)

    page_results = []
    for page_idx in range(total_pages):
        regions = pages_map.get(page_idx, [])
        regions.sort(key=lambda r: r.region_index)

        # Combine region texts into full page text
        page_text_parts = []
        region_dicts = []

        for r in regions:
            text = r.text.strip()
            if not text or text == "[OCR_ERROR: all retries exhausted]":
                continue

            page_text_parts.append(text)

            region_dict: dict[str, Any] = {
                "index": r.region_index,
                "label": r.label,
                "content": text,
            }
            if include_bboxes and r.bbox:
                region_dict["bbox_2d"] = r.bbox
            if r.latency_ms > 0:
                region_dict["latency_ms"] = round(r.latency_ms, 1)

            region_dicts.append(region_dict)

        raw_text = "\n\n".join(page_text_parts)

        page_results.append(
            PageResult(
                page_number=page_idx + 1,
                raw_text=raw_text,
                regions=region_dicts,
            )
        )

    return page_results


def build_document_result(
    filename: str,
    page_results: list[PageResult],
    processing_time: float,
    file_type: str = "pdf",
) -> DocumentResult:
    """Build the final DocumentResult with metadata."""
    total_pages = len(page_results)
    pps = total_pages / processing_time if processing_time > 0 else 0

    return DocumentResult(
        filename=filename,
        total_pages=total_pages,
        pages=page_results,
        processing_time_sec=round(processing_time, 3),
        pages_per_second=round(pps, 2),
        file_type=file_type,
    )


def to_markdown(doc: DocumentResult) -> str:
    """Convert DocumentResult to Markdown string."""
    parts = []

    # Header
    parts.append(f"# {doc.filename}\n")
    parts.append(
        f"> Processed {doc.total_pages} pages in {doc.processing_time_sec}s "
        f"({doc.pages_per_second} pages/sec)\n"
    )

    for page in doc.pages:
        parts.append(f"\n---\n\n<!-- Page {page.page_number} -->\n")

        if page.raw_text.strip():
            parts.append(page.raw_text)
        else:
            parts.append("*[Empty page]*")

        parts.append("")  # trailing newline

    return "\n".join(parts)


def to_json(doc: DocumentResult) -> dict:
    """Convert DocumentResult to JSON-serializable dict."""
    return {
        "metadata": {
            "filename": doc.filename,
            "file_type": doc.file_type,
            "total_pages": doc.total_pages,
            "processing_time_sec": doc.processing_time_sec,
            "pages_per_second": doc.pages_per_second,
        },
        "pages": [
            {
                "page_number": p.page_number,
                "raw_text": p.raw_text,
                "regions": p.regions,
            }
            for p in doc.pages
        ],
    }


def save_outputs(
    doc: DocumentResult,
    output_dir: str | Path,
    output_format: str = "both",
) -> dict[str, Path]:
    """
    Save Markdown and/or JSON output files.

    Returns dict mapping format name to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(doc.filename).stem
    saved = {}

    if output_format in ("markdown", "both"):
        md_path = output_dir / f"{stem}.md"
        md_content = to_markdown(doc)
        md_path.write_text(md_content, encoding="utf-8")
        saved["markdown"] = md_path
        logger.info("Saved Markdown → %s (%d chars)", md_path, len(md_content))

    if output_format in ("json", "both"):
        json_path = output_dir / f"{stem}.json"
        json_content = to_json(doc)
        json_path.write_text(
            json.dumps(json_content, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved["json"] = json_path
        logger.info("Saved JSON → %s", json_path)

    return saved
