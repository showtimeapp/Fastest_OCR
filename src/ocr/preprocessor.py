"""
Image Preprocessor — resize and prepare images for GLM-OCR.

Handles pixel budget management to stay within the model's
min_pixels / max_pixels constraints while preserving aspect ratio.
"""

from __future__ import annotations

import math
from PIL import Image


def resize_for_model(
    image: Image.Image,
    min_pixels: int = 12544,
    max_pixels: int = 71372800,
) -> Image.Image:
    """
    Resize image to fit within the model's pixel budget.

    - If total pixels < min_pixels → upscale
    - If total pixels > max_pixels → downscale
    - Preserves aspect ratio
    """
    w, h = image.size
    total = w * h

    if min_pixels <= total <= max_pixels:
        return image

    if total > max_pixels:
        scale = math.sqrt(max_pixels / total)
    else:
        scale = math.sqrt(min_pixels / total)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    return image.resize((new_w, new_h), Image.LANCZOS)


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert image to RGB if necessary (handles RGBA, L, P modes)."""
    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        # Composite on white background
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    return image.convert("RGB")


def preprocess_image(
    image: Image.Image,
    min_pixels: int = 12544,
    max_pixels: int = 71372800,
) -> Image.Image:
    """Full preprocessing pipeline: RGB conversion + resize."""
    image = ensure_rgb(image)
    image = resize_for_model(image, min_pixels, max_pixels)
    return image
