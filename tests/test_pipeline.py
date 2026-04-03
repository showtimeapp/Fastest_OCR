"""
Tests for GLM-OCR Production Service.

Run: python -m pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, AppConfig
from src.router import detect_file_type
from src.handlers.text_handler import process_text_file
from src.ocr.postprocessor import (
    OCRResult,
    assemble_page_results,
    build_document_result,
    to_json,
    to_markdown,
)
from src.ocr.layout import (
    DetectedRegion,
    filter_ocr_regions,
    get_prompt_for_label,
)
from src.ocr.preprocessor import preprocess_image, ensure_rgb


# ============================================================
# Config Tests
# ============================================================

class TestConfig:
    def test_default_config(self):
        config = AppConfig()
        assert config.backend.type == "vllm"
        assert config.backend.port == 8080
        assert config.model.name == "zai-org/GLM-OCR"
        assert config.pdf.render_dpi == 300
        assert config.output.format == "both"

    def test_load_missing_config(self):
        config = load_config("/nonexistent/path.yaml")
        assert config.backend.type == "vllm"


# ============================================================
# Router Tests
# ============================================================

class TestRouter:
    def test_detect_pdf(self):
        assert detect_file_type("document.pdf") == "pdf"
        assert detect_file_type("REPORT.PDF") == "pdf"

    def test_detect_text(self):
        assert detect_file_type("notes.txt") == "text"
        assert detect_file_type("readme.md") == "text"

    def test_detect_doc(self):
        assert detect_file_type("report.docx") == "doc"
        assert detect_file_type("old.doc") == "doc"

    def test_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported"):
            detect_file_type("image.png")


# ============================================================
# Text Handler Tests
# ============================================================

class TestTextHandler:
    def test_process_text(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello World\nLine 2")

        result = process_text_file(txt_file)
        assert result.filename == "test.txt"
        assert result.total_pages == 1
        assert result.file_type == "txt"
        assert "Hello World" in result.pages[0].raw_text


# ============================================================
# Post-Processor Tests
# ============================================================

class TestPostProcessor:
    def _make_results(self):
        return [
            OCRResult(text="Title text", page_index=0, region_index=0, label="title"),
            OCRResult(text="Body paragraph", page_index=0, region_index=1, label="text"),
            OCRResult(text="Page 2 content", page_index=1, region_index=0, label="text"),
        ]

    def test_assemble_pages(self):
        results = self._make_results()
        pages = assemble_page_results(results, total_pages=2)
        assert len(pages) == 2
        assert "Title text" in pages[0].raw_text
        assert "Body paragraph" in pages[0].raw_text
        assert "Page 2 content" in pages[1].raw_text

    def test_to_markdown(self):
        results = self._make_results()
        pages = assemble_page_results(results, total_pages=2)
        doc = build_document_result("test.pdf", pages, processing_time=1.5)
        md = to_markdown(doc)
        assert "# test.pdf" in md
        assert "Title text" in md
        assert "Page 2" in md

    def test_to_json(self):
        results = self._make_results()
        pages = assemble_page_results(results, total_pages=2)
        doc = build_document_result("test.pdf", pages, processing_time=1.5)
        j = to_json(doc)
        assert j["metadata"]["total_pages"] == 2
        assert j["metadata"]["filename"] == "test.pdf"
        assert len(j["pages"]) == 2

    def test_empty_page(self):
        results = [OCRResult(text="Only page 2", page_index=1, region_index=0)]
        pages = assemble_page_results(results, total_pages=2)
        assert pages[0].raw_text == ""
        assert "Only page 2" in pages[1].raw_text


# ============================================================
# Layout Tests
# ============================================================

class TestLayout:
    def test_filter_regions(self):
        regions = [
            DetectedRegion(bbox=[0, 0, 100, 100], label="text", confidence=0.9, label_id=0),
            DetectedRegion(bbox=[0, 0, 100, 100], label="figure", confidence=0.8, label_id=2),
            DetectedRegion(bbox=[0, 0, 100, 100], label="table", confidence=0.85, label_id=4),
        ]
        filtered = filter_ocr_regions(regions)
        assert len(filtered) == 2
        labels = {r.label for r in filtered}
        assert "text" in labels
        assert "table" in labels
        assert "figure" not in labels

    def test_prompt_mapping(self):
        assert get_prompt_for_label("text") == "Text Recognition:"
        assert get_prompt_for_label("table") == "Table Recognition:"
        assert get_prompt_for_label("equation") == "Formula Recognition:"
        assert get_prompt_for_label("unknown") == "Text Recognition:"


# ============================================================
# Preprocessor Tests
# ============================================================

class TestPreprocessor:
    def test_ensure_rgb(self):
        from PIL import Image
        rgba = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        rgb = ensure_rgb(rgba)
        assert rgb.mode == "RGB"

    def test_preprocess_small_image(self):
        from PIL import Image
        tiny = Image.new("RGB", (10, 10))
        result = preprocess_image(tiny, min_pixels=12544)
        w, h = result.size
        assert w * h >= 12544


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
