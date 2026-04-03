"""
Layout Detector — PP-DocLayout-V3 via ONNX Runtime.

Detects document regions (text, table, formula, figure, title, etc.)
and returns bounding boxes + labels for parallel OCR processing.

If layout detection is disabled or the model is not available,
falls back to full-page OCR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# PP-DocLayout-V3 label map
LAYOUT_LABELS = {
    0: "text",
    1: "title",
    2: "figure",
    3: "figure_caption",
    4: "table",
    5: "table_caption",
    6: "header",
    7: "footer",
    8: "reference",
    9: "equation",
    10: "list",
    11: "code",
    12: "seal",
}

# Which labels need OCR (skip figures, headers, footers for speed)
OCR_LABELS = {"text", "title", "table", "equation", "list", "code", "reference", "seal", "table_caption", "figure_caption"}

# Prompt mapping for each region type
LABEL_PROMPTS = {
    "text": "Text Recognition:",
    "title": "Text Recognition:",
    "table": "Table Recognition:",
    "equation": "Formula Recognition:",
    "list": "Text Recognition:",
    "code": "Text Recognition:",
    "reference": "Text Recognition:",
    "seal": "Text Recognition:",
    "table_caption": "Text Recognition:",
    "figure_caption": "Text Recognition:",
}


@dataclass
class DetectedRegion:
    """A detected layout region on a page."""
    bbox: list[float]  # [x1, y1, x2, y2] in pixel coords
    label: str
    confidence: float
    label_id: int


@dataclass
class LayoutDetector:
    """
    PP-DocLayout-V3 layout detector.

    Uses ONNX Runtime for GPU-accelerated inference.
    Falls back gracefully if the model is not available.
    """

    model_path: Optional[str] = None
    confidence_threshold: float = 0.5
    _session: Optional[object] = None
    _available: bool = False

    def load(self):
        """Load the ONNX model."""
        try:
            import onnxruntime as ort

            if self.model_path and Path(self.model_path).exists():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self._session = ort.InferenceSession(
                    self.model_path, providers=providers
                )
                self._available = True
                logger.info("Layout detector loaded from %s", self.model_path)
            else:
                logger.warning(
                    "Layout model not found at %s — using glmocr SDK layout or full-page fallback",
                    self.model_path,
                )
        except ImportError:
            logger.warning("onnxruntime not installed — layout detection disabled")
        except Exception as exc:
            logger.warning("Failed to load layout model: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, image: Image.Image) -> list[DetectedRegion]:
        """
        Detect layout regions in a page image.

        If the model is not loaded, returns a single full-page region
        so the pipeline can proceed without layout analysis.
        """
        if not self._available:
            return self._full_page_fallback(image)

        try:
            return self._run_inference(image)
        except Exception as exc:
            logger.warning("Layout detection failed: %s — falling back to full page", exc)
            return self._full_page_fallback(image)

    def _full_page_fallback(self, image: Image.Image) -> list[DetectedRegion]:
        """Return a single region covering the entire page."""
        w, h = image.size
        return [
            DetectedRegion(
                bbox=[0, 0, w, h],
                label="text",
                confidence=1.0,
                label_id=0,
            )
        ]

    def _run_inference(self, image: Image.Image) -> list[DetectedRegion]:
        """Run the ONNX layout model."""
        # Preprocess: resize to model input size
        input_size = (640, 640)
        orig_w, orig_h = image.size

        img_resized = image.resize(input_size, Image.BILINEAR)
        img_array = np.array(img_resized).astype(np.float32)

        # Normalize [0, 255] → [0, 1]
        img_array = img_array / 255.0

        # HWC → CHW → NCHW
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        # Run inference
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: img_array})

        # Parse outputs (format depends on the model export)
        # Typical: boxes [N, 4], scores [N], labels [N]
        regions = []
        if len(outputs) >= 3:
            boxes, scores, label_ids = outputs[0], outputs[1], outputs[2]

            scale_x = orig_w / input_size[0]
            scale_y = orig_h / input_size[1]

            for i in range(len(scores)):
                if scores[i] < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = boxes[i]
                bbox = [
                    float(x1 * scale_x),
                    float(y1 * scale_y),
                    float(x2 * scale_x),
                    float(y2 * scale_y),
                ]

                lid = int(label_ids[i])
                label = LAYOUT_LABELS.get(lid, "text")

                regions.append(
                    DetectedRegion(
                        bbox=bbox,
                        label=label,
                        confidence=float(scores[i]),
                        label_id=lid,
                    )
                )

        if not regions:
            return self._full_page_fallback(image)

        # Sort top-to-bottom, then left-to-right (reading order)
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

        logger.debug("Detected %d layout regions", len(regions))
        return regions


def crop_region(image: Image.Image, bbox: list[float], padding: int = 5) -> Image.Image:
    """Crop a region from an image with optional padding."""
    w, h = image.size
    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(w, int(bbox[2]) + padding)
    y2 = min(h, int(bbox[3]) + padding)
    return image.crop((x1, y1, x2, y2))


def filter_ocr_regions(regions: list[DetectedRegion]) -> list[DetectedRegion]:
    """Filter regions that need OCR processing."""
    return [r for r in regions if r.label in OCR_LABELS]


def get_prompt_for_label(label: str) -> str:
    """Get the appropriate OCR prompt for a region label."""
    return LABEL_PROMPTS.get(label, "Text Recognition:")
