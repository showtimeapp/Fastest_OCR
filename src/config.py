"""
Configuration loader — reads config.yaml into typed Pydantic models.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class BackendConfig(BaseModel):
    type: Literal["vllm", "sglang"] = "vllm"
    host: str = "localhost"
    port: int = 8080
    api_key: Optional[str] = None
    connect_timeout: int = 300
    request_timeout: int = 300
    max_concurrent_requests: int = 16


class ModelConfig(BaseModel):
    name: str = "zai-org/GLM-OCR"
    dtype: Literal["float16", "bfloat16"] = "float16"
    max_new_tokens: int = 8192
    temperature: float = 0.01


class PDFConfig(BaseModel):
    render_dpi: int = 300
    rasterize_workers: int = 8
    image_format: Literal["JPEG", "PNG"] = "JPEG"
    jpeg_quality: int = 95
    max_pixels: int = 71372800
    min_pixels: int = 12544
    enable_layout: bool = True
    region_batch_size: int = 8
    save_region_images: bool = False


class DocConfig(BaseModel):
    preserve_images: bool = False


class OutputConfig(BaseModel):
    format: Literal["json", "markdown", "both"] = "both"
    output_dir: str = "./output"
    include_page_metadata: bool = True
    include_bboxes: bool = True


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_upload_size_mb: int = 500


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


class AppConfig(BaseModel):
    backend: BackendConfig = Field(default_factory=BackendConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    pdf: PDFConfig = Field(default_factory=PDFConfig)
    doc: DocConfig = Field(default_factory=DocConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load config from YAML file, falling back to defaults."""
    if config_path is None:
        config_path = os.environ.get(
            "GLM_OCR_CONFIG",
            Path(__file__).parent.parent / "config.yaml",
        )

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        return AppConfig(**raw)

    return AppConfig()
