"""
Backend Comparison — benchmark vLLM vs SGLang on the same PDF.

Prerequisites:
  - vLLM running on port 8080
  - SGLang running on port 8081

Usage:
  python -m benchmark.compare_backends test.pdf
  python -m benchmark.compare_backends test.pdf --iterations 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from benchmark.benchmark import run_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


async def compare(pdf_path: str, iterations: int = 3):
    """Run benchmark on both backends and compare."""
    config = load_config()

    # --- vLLM ---
    logger.info("\n" + "=" * 70)
    logger.info("BACKEND: vLLM (port 8080)")
    logger.info("=" * 70)
    config_vllm = config.model_copy(deep=True)
    config_vllm.backend.type = "vllm"
    config_vllm.backend.port = 8080
    vllm_results = await run_benchmark(pdf_path, config_vllm, iterations=iterations)

    # --- SGLang ---
    logger.info("\n" + "=" * 70)
    logger.info("BACKEND: SGLang (port 8081)")
    logger.info("=" * 70)
    config_sglang = config.model_copy(deep=True)
    config_sglang.backend.type = "sglang"
    config_sglang.backend.port = 8081
    sglang_results = await run_benchmark(pdf_path, config_sglang, iterations=iterations)

    # --- Final comparison ---
    logger.info("\n" + "=" * 70)
    logger.info("FINAL: vLLM vs SGLang")
    logger.info("=" * 70)

    vllm_pps = vllm_results[0]["avg_pages_per_sec"]
    sglang_pps = sglang_results[0]["avg_pages_per_sec"]

    logger.info("  vLLM:   %.2f pages/sec", vllm_pps)
    logger.info("  SGLang: %.2f pages/sec", sglang_pps)

    if sglang_pps > vllm_pps:
        logger.info("  → SGLang is %.1f%% faster", ((sglang_pps / vllm_pps) - 1) * 100)
    else:
        logger.info("  → vLLM is %.1f%% faster", ((vllm_pps / sglang_pps) - 1) * 100)


def main():
    parser = argparse.ArgumentParser(description="Compare vLLM vs SGLang")
    parser.add_argument("pdf", help="PDF file to benchmark")
    parser.add_argument("-n", "--iterations", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(compare(args.pdf, args.iterations))


if __name__ == "__main__":
    main()
