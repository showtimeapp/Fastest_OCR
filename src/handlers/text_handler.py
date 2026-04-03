"""
Text Handler — process .txt files directly (no OCR needed).
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.ocr.postprocessor import (
    DocumentResult,
    PageResult,
    build_document_result,
)

logger = logging.getLogger(__name__)


def process_text_file(file_path: str | Path) -> DocumentResult:
    """
    Read a .txt file and wrap it in DocumentResult.
    No OCR involved — direct text extraction.
    """
    file_path = Path(file_path)
    content = file_path.read_text(encoding="utf-8", errors="replace")

    page_results = [
        PageResult(
            page_number=1,
            raw_text=content,
            regions=[
                {
                    "index": 0,
                    "label": "text",
                    "content": content,
                }
            ],
        )
    ]

    doc_result = build_document_result(
        filename=file_path.name,
        page_results=page_results,
        processing_time=0.001,
        file_type="txt",
    )

    logger.info("Text file processed: %s (%d chars)", file_path.name, len(content))
    return doc_result
