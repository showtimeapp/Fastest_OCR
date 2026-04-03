"""
OCR Engine — async client using OpenAI SDK for max speed.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from openai import AsyncOpenAI

from src.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    page_index: int = 0
    region_index: int = 0
    label: str = "full_page"
    bbox: Optional[list[float]] = None
    latency_ms: float = 0.0


@dataclass
class OCREngine:
    config: AppConfig
    _client: Optional[AsyncOpenAI] = field(default=None, init=False, repr=False)

    async def start(self):
        self._client = AsyncOpenAI(
            base_url=f"http://{self.config.backend.host}:{self.config.backend.port}/v1",
            api_key="dummy",
            timeout=self.config.backend.request_timeout,
            max_retries=3,
        )
        logger.info("OCR engine started → %s:%d", self.config.backend.host, self.config.backend.port)

    async def stop(self):
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("OCR engine stopped")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()

    def _encode_image(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    async def ocr_single(
        self,
        image: Image.Image,
        prompt: str = "Text Recognition:",
        page_index: int = 0,
        region_index: int = 0,
        label: str = "full_page",
        bbox: Optional[list[float]] = None,
        **kwargs,
    ) -> OCRResult:
        b64 = self._encode_image(image)
        t0 = time.perf_counter()

        try:
            resp = await self._client.chat.completions.create(
                model=self.config.model.name,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ]}],
                max_tokens=self.config.model.max_new_tokens,
                temperature=self.config.model.temperature,
            )
            text = resp.choices[0].message.content
            latency = (time.perf_counter() - t0) * 1000

            return OCRResult(
                text=text,
                page_index=page_index,
                region_index=region_index,
                label=label,
                bbox=bbox,
                latency_ms=latency,
            )
        except Exception as exc:
            logger.error("OCR failed page=%d: %s", page_index, str(exc)[:200])
            return OCRResult(
                text="",
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
        n = len(images)
        prompts = prompts or ["Text Recognition:"] * n
        page_indices = page_indices or list(range(n))
        region_indices = region_indices or [0] * n
        labels = labels or ["full_page"] * n
        bboxes = bboxes or [None] * n

        tasks = [
            self.ocr_single(images[i], prompts[i], page_indices[i], region_indices[i], labels[i], bboxes[i])
            for i in range(n)
        ]
        return list(await asyncio.gather(*tasks))

    async def health_check(self) -> bool:
        try:
            models = await self._client.models.list()
            return True
        except Exception:
            return False
        