"""
File Router — detects file type and dispatches to the correct handler.

Routing:
  .txt         → TextHandler (no OCR)
  .doc / .docx → DocHandler (no OCR)
  .pdf         → PDFHandler (OCR pipeline)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.config import AppConfig
from src.handlers.doc_handler import process_doc_file
from src.handlers.pdf_handler import process_pdf
from src.handlers.text_handler import process_text_file
from src.ocr.engine import OCREngine
from src.ocr.layout import LayoutDetector
from src.ocr.postprocessor import DocumentResult, save_outputs

logger = logging.getLogger(__name__)

# Supported file extensions
TEXT_EXTENSIONS = {".txt", ".text", ".md", ".markdown", ".rst"}
DOC_EXTENSIONS = {".doc", ".docx"}
PDF_EXTENSIONS = {".pdf"}

SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | DOC_EXTENSIONS | PDF_EXTENSIONS


def detect_file_type(file_path: str | Path) -> str:
    """Detect the file type category from extension."""
    suffix = Path(file_path).suffix.lower()

    if suffix in TEXT_EXTENSIONS:
        return "text"
    elif suffix in DOC_EXTENSIONS:
        return "doc"
    elif suffix in PDF_EXTENSIONS:
        return "pdf"
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


async def process_file(
    file_path: str | Path,
    engine: OCREngine,
    config: AppConfig,
    layout_detector: Optional[LayoutDetector] = None,
    output_dir: Optional[str | Path] = None,
) -> tuple[DocumentResult, dict[str, Path]]:
    """
    Process a single file end-to-end.

    1. Detect file type
    2. Route to appropriate handler
    3. Save outputs (Markdown + JSON)

    Returns (DocumentResult, {format: filepath})
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_type = detect_file_type(file_path)
    logger.info("Processing %s (type=%s)", file_path.name, file_type)

    # --- Route to handler ---
    if file_type == "text":
        doc_result = process_text_file(file_path)

    elif file_type == "doc":
        doc_result = process_doc_file(file_path)

    elif file_type == "pdf":
        doc_result = await process_pdf(
            pdf_path=file_path,
            engine=engine,
            config=config,
            layout_detector=layout_detector,
        )

    else:
        raise ValueError(f"Unhandled file type: {file_type}")

    # --- Save outputs ---
    out_dir = output_dir or config.output.output_dir
    saved = save_outputs(
        doc=doc_result,
        output_dir=out_dir,
        output_format=config.output.format,
    )

    return doc_result, saved


async def process_files(
    file_paths: list[str | Path],
    engine: OCREngine,
    config: AppConfig,
    layout_detector: Optional[LayoutDetector] = None,
    output_dir: Optional[str | Path] = None,
) -> list[tuple[DocumentResult, dict[str, Path]]]:
    """Process multiple files sequentially."""
    results = []
    for fp in file_paths:
        try:
            result = await process_file(
                file_path=fp,
                engine=engine,
                config=config,
                layout_detector=layout_detector,
                output_dir=output_dir,
            )
            results.append(result)
        except Exception as exc:
            logger.error("Failed to process %s: %s", fp, exc)
    return results
