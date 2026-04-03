"""
GLM-OCR Production Service — CLI Entry Point.

Usage:
  # Process a single file
  python -m src.main input.pdf

  # Process multiple files
  python -m src.main file1.pdf file2.docx file3.txt

  # Process a directory
  python -m src.main ./documents/

  # Specify output directory
  python -m src.main input.pdf --output ./results/

  # Use custom config
  python -m src.main input.pdf --config my_config.yaml

  # Enable debug logging
  python -m src.main input.pdf --log-level DEBUG
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from src.config import load_config
from src.ocr.engine import OCREngine
from src.ocr.layout import LayoutDetector
from src.router import SUPPORTED_EXTENSIONS, process_file, process_files


def setup_logging(level: str, fmt: str):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        stream=sys.stderr,
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def collect_files(paths: list[str]) -> list[Path]:
    """Expand directories and collect all supported files."""
    files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                files.extend(sorted(path.glob(f"*{ext}")))
        elif path.is_file():
            files.append(path)
        else:
            logging.getLogger(__name__).warning("Skipping %s (not found)", p)
    return files


async def run(args: argparse.Namespace):
    """Main async entry point."""
    config = load_config(args.config)

    # Override log level from CLI
    if args.log_level:
        config.logging.level = args.log_level
    setup_logging(config.logging.level, config.logging.format)

    logger = logging.getLogger(__name__)

    # Override output dir from CLI
    if args.output:
        config.output.output_dir = args.output

    # Collect input files
    files = collect_files(args.input)
    if not files:
        logger.error("No supported files found. Supported: %s", ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        sys.exit(1)

    logger.info("Found %d file(s) to process", len(files))

    # Initialize layout detector
    layout_detector = LayoutDetector(model_path=args.layout_model)
    if config.pdf.enable_layout:
        layout_detector.load()

    # Initialize OCR engine and process
    engine = OCREngine(config=config)
    async with engine:
        # Health check
        healthy = await engine.health_check()
        if not healthy:
            logger.error(
                "Backend not reachable at %s:%d — is vLLM/SGLang running?",
                config.backend.host,
                config.backend.port,
            )
            sys.exit(1)
        logger.info("Backend health check passed ✓")

        t0 = time.perf_counter()
        results = await process_files(
            file_paths=files,
            engine=engine,
            config=config,
            layout_detector=layout_detector,
            output_dir=config.output.output_dir,
        )
        total_time = time.perf_counter() - t0

    # Summary
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    total_pages = 0
    for doc_result, saved_files in results:
        total_pages += doc_result.total_pages
        logger.info(
            "  %s → %d pages, %.2fs (%.2f pg/s)",
            doc_result.filename,
            doc_result.total_pages,
            doc_result.processing_time_sec,
            doc_result.pages_per_second,
        )
        for fmt, path in saved_files.items():
            logger.info("    %s: %s", fmt, path)

    logger.info("-" * 60)
    logger.info(
        "Total: %d files, %d pages, %.2fs (%.2f pages/sec overall)",
        len(results),
        total_pages,
        total_time,
        total_pages / total_time if total_time > 0 else 0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="GLM-OCR Production Service — process documents with OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Input file(s) or directory to process",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: from config.yaml)",
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (overrides config)",
    )
    parser.add_argument(
        "--layout-model",
        default=None,
        help="Path to PP-DocLayout-V3 ONNX model file",
    )

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
