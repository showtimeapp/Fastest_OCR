"""
OCR Engine — async client for vLLM / SGLang OpenAI-compatible API.

Handles:
  - Single image OCR
  - Batch OCR with concurrency control
  - Base64 image encoding
  - Retry logic with exponential backoff
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
from PIL import Image

from src.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from a single OCR call."""
    text: str
    page_index: int = 0
    region_index: int = 0
    label: str = "full_page"
    bbox: Optional[list[float]] = None
    latency_ms: float = 0.0


@dataclass
class OCREngine:
    """Async OCR client that sends images to vLLM/SGLang."""

    config: AppConfig
    _client: Optional[httpx.AsyncClient] = field(default=None, init=False, repr=False)
    _semaphore: Optional[asyncio.Semaphore] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._semaphore = asyncio.Semaphore(self.config.backend.max_concurrent_requests)

    @property
    def base_url(self) -> str:
        return f"http://{self.config.backend.host}:{self.config.backend.port}"

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    async def start(self):
        """Initialize the HTTP client."""
        headers = {"Content-Type": "application/json"}
        if self.config.backend.api_key:
            headers["Authorization"] = f"Bearer {self.config.backend.api_key}"

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.config.backend.connect_timeout,
                read=self.config.backend.request_timeout,
                write=30.0,
                pool=30.0,
            ),
            headers=headers,
            limits=httpx.Limits(
                max_connections=self.config.backend.max_concurrent_requests + 4,
                max_keepalive_connections=self.config.backend.max_concurrent_requests,
            ),
        )
        logger.info("OCR engine client started → %s", self.api_url)

    async def stop(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("OCR engine client stopped")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 data URI."""
        buf = io.BytesIO()
        fmt = self.config.pdf.image_format
        save_kwargs = {"format": fmt}
        if fmt == "JPEG":
            save_kwargs["quality"] = self.config.pdf.jpeg_quality
        image.save(buf, **save_kwargs)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "image/jpeg" if fmt == "JPEG" else "image/png"
        return f"data:{mime};base64,{b64}"

    def _build_payload(
        self,
        image: Image.Image,
        prompt: str = "Text Recognition:",
    ) -> dict:
        """Build the OpenAI-compatible chat completion payload."""
        data_uri = self._encode_image(image)
        return {
            "model": self.config.model.name,
            "max_tokens": self.config.model.max_new_tokens,
            "temperature": self.config.model.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

    async def ocr_single(
        self,
        image: Image.Image,
        prompt: str = "Text Recognition:",
        page_index: int = 0,
        region_index: int = 0,
        label: str = "full_page",
        bbox: Optional[list[float]] = None,
        max_retries: int = 3,
    ) -> OCRResult:
        """
        Run OCR on a single image with concurrency control and retries.
        """
        payload = self._build_payload(image, prompt)

        for attempt in range(max_retries):
            async with self._semaphore:
                t0 = time.perf_counter()
                try:
                    resp = await self._client.post(self.api_url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"]
                    latency = (time.perf_counter() - t0) * 1000

                    logger.debug(
                        "OCR page=%d region=%d label=%s latency=%.0fms tokens=%d",
                        page_index,
                        region_index,
                        label,
                        latency,
                        len(text.split()),
                    )

                    return OCRResult(
                        text=text,
                        page_index=page_index,
                        region_index=region_index,
                        label=label,
                        bbox=bbox,
                        latency_ms=latency,
                    )

                except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                    wait = 2 ** attempt
                    logger.warning(
                        "OCR attempt %d/%d failed (page=%d region=%d): %s — retrying in %ds",
                        attempt + 1,
                        max_retries,
                        page_index,
                        region_index,
                        str(exc)[:200],
                        wait,
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait)

        # All retries exhausted
        logger.error("OCR failed after %d retries (page=%d region=%d)", max_retries, page_index, region_index)
        return OCRResult(
            text="[OCR_ERROR: all retries exhausted]",
            page_index=page_index,
            region_index=region_index,
            label=label,
            bbox=bbox,
        )

    async def ocr_batch(
        self,
        images: list[Image.Image],
        prompts: Optional[list[str]] = None,
        page_indices: Optional[list[int]] = None,
        region_indices: Optional[list[int]] = None,
        labels: Optional[list[str]] = None,
        bboxes: Optional[list[Optional[list[float]]]] = None,
    ) -> list[OCRResult]:
        """
        Run OCR on a batch of images concurrently.
        Concurrency is bounded by self._semaphore.
        """
        n = len(images)
        prompts = prompts or ["Text Recognition:"] * n
        page_indices = page_indices or list(range(n))
        region_indices = region_indices or [0] * n
        labels = labels or ["full_page"] * n
        bboxes = bboxes or [None] * n

        tasks = [
            self.ocr_single(
                image=images[i],
                prompt=prompts[i],
                page_index=page_indices[i],
                region_index=region_indices[i],
                label=labels[i],
                bbox=bboxes[i],
            )
            for i in range(n)
        ]

        results = await asyncio.gather(*tasks)
        return list(results)

    async def health_check(self) -> bool:
        """Check if the backend is responsive."""
        try:
            resp = await self._client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except Exception:
            # Try models endpoint (vLLM style)
            try:
                resp = await self._client.get(f"{self.base_url}/v1/models")
                return resp.status_code == 200
            except Exception:
                return False
