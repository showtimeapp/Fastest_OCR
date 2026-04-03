"""
Benchmark Tool — measure OCR throughput and latency.

Usage:
  # Benchmark a single PDF
  python -m benchmark.benchmark test.pdf

  # Benchmark with specific config
  python -m benchmark.benchmark test.pdf --config config.yaml

  # Compare layout ON vs OFF
  python -m benchmark.benchmark test.pdf --compare-layout

  # Run N iterations for statistical significance
  python -m benchmark.benchmark test.pdf --iterations 5

  # Benchmark specific batch sizes
  python -m benchmark.benchmark test.pdf --batch-sizes 1,4,8,16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, AppConfig
from src.ocr.engine import OCREngine
from src.ocr.layout import LayoutDetector
from src.handlers.pdf_handler import process_pdf, get_pdf_page_count


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


async def benchmark_single(
    pdf_path: str,
    config: AppConfig,
    layout_detector: LayoutDetector,
    iteration: int = 1,
) -> dict:
    """Run a single benchmark iteration."""
    engine = OCREngine(config=config)
    async with engine:
        healthy = await engine.health_check()
        if not healthy:
            raise RuntimeError(f"Backend not reachable at {config.backend.host}:{config.backend.port}")

        t0 = time.perf_counter()
        doc_result = await process_pdf(
            pdf_path=pdf_path,
            engine=engine,
            config=config,
            layout_detector=layout_detector,
        )
        elapsed = time.perf_counter() - t0

    num_pages = doc_result.total_pages
    latencies = []
    for page in doc_result.pages:
        for region in page.regions:
            if "latency_ms" in region:
                latencies.append(region["latency_ms"])

    result = {
        "iteration": iteration,
        "total_pages": num_pages,
        "total_time_sec": round(elapsed, 3),
        "pages_per_sec": round(num_pages / elapsed, 3) if elapsed > 0 else 0,
        "layout_enabled": config.pdf.enable_layout,
        "total_ocr_calls": len(latencies),
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0,
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 1),
        "max_latency_ms": round(max(latencies), 1) if latencies else 0,
    }
    return result


async def run_benchmark(
    pdf_path: str,
    config: AppConfig,
    iterations: int = 3,
    compare_layout: bool = False,
    batch_sizes: list[int] | None = None,
):
    """Run full benchmark suite."""
    num_pages = get_pdf_page_count(pdf_path)
    logger.info("=" * 70)
    logger.info("BENCHMARK: %s (%d pages)", Path(pdf_path).name, num_pages)
    logger.info("Backend: %s @ %s:%d", config.backend.type, config.backend.host, config.backend.port)
    logger.info("Iterations: %d", iterations)
    logger.info("=" * 70)

    all_results = []

    # --- Test different configurations ---
    configs_to_test = []

    if compare_layout:
        # Layout OFF
        cfg_off = config.model_copy(deep=True)
        cfg_off.pdf.enable_layout = False
        configs_to_test.append(("layout=OFF", cfg_off))

        # Layout ON
        cfg_on = config.model_copy(deep=True)
        cfg_on.pdf.enable_layout = True
        configs_to_test.append(("layout=ON", cfg_on))
    elif batch_sizes:
        for bs in batch_sizes:
            cfg = config.model_copy(deep=True)
            cfg.pdf.region_batch_size = bs
            cfg.backend.max_concurrent_requests = bs
            configs_to_test.append((f"concurrency={bs}", cfg))
    else:
        configs_to_test.append(("default", config))

    for label, cfg in configs_to_test:
        logger.info("")
        logger.info("--- Config: %s ---", label)

        layout_det = LayoutDetector()
        if cfg.pdf.enable_layout:
            layout_det.load()

        run_results = []
        for i in range(iterations):
            # Warmup: skip first iteration in stats if > 1
            result = await benchmark_single(pdf_path, cfg, layout_det, iteration=i + 1)
            run_results.append(result)
            logger.info(
                "  Iteration %d: %.2f pages/sec (%.2fs total, %d OCR calls, avg %.0fms latency)",
                i + 1,
                result["pages_per_sec"],
                result["total_time_sec"],
                result["total_ocr_calls"],
                result["avg_latency_ms"],
            )

        # Aggregate stats (skip first iteration as warmup if multiple iterations)
        stats_runs = run_results[1:] if len(run_results) > 1 else run_results
        pps_values = [r["pages_per_sec"] for r in stats_runs]

        summary = {
            "config": label,
            "pages": num_pages,
            "iterations": len(stats_runs),
            "avg_pages_per_sec": round(statistics.mean(pps_values), 3),
            "std_pages_per_sec": round(statistics.stdev(pps_values), 3) if len(pps_values) > 1 else 0,
            "min_pages_per_sec": round(min(pps_values), 3),
            "max_pages_per_sec": round(max(pps_values), 3),
            "avg_ocr_calls": round(statistics.mean([r["total_ocr_calls"] for r in stats_runs])),
            "avg_latency_ms": round(statistics.mean([r["avg_latency_ms"] for r in stats_runs]), 1),
            "runs": run_results,
        }
        all_results.append(summary)

        logger.info(
            "  SUMMARY: %.2f ± %.3f pages/sec (min=%.2f, max=%.2f)",
            summary["avg_pages_per_sec"],
            summary["std_pages_per_sec"],
            summary["min_pages_per_sec"],
            summary["max_pages_per_sec"],
        )

    # --- Final comparison ---
    if len(all_results) > 1:
        logger.info("")
        logger.info("=" * 70)
        logger.info("COMPARISON")
        logger.info("=" * 70)
        baseline = all_results[0]["avg_pages_per_sec"]
        for r in all_results:
            speedup = r["avg_pages_per_sec"] / baseline if baseline > 0 else 0
            logger.info(
                "  %-20s  %.2f pages/sec  (%.1fx vs baseline)",
                r["config"],
                r["avg_pages_per_sec"],
                speedup,
            )

    # Save results
    output_path = Path("benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nResults saved to %s", output_path)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="GLM-OCR Benchmark Tool")
    parser.add_argument("pdf", help="PDF file to benchmark")
    parser.add_argument("-c", "--config", default=None, help="Config YAML path")
    parser.add_argument("-n", "--iterations", type=int, default=3, help="Number of iterations")
    parser.add_argument("--compare-layout", action="store_true", help="Compare layout ON vs OFF")
    parser.add_argument(
        "--batch-sizes",
        default=None,
        help="Comma-separated batch sizes to test (e.g., 1,4,8,16)",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    batch_sizes = None
    if args.batch_sizes:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    asyncio.run(
        run_benchmark(
            pdf_path=args.pdf,
            config=config,
            iterations=args.iterations,
            compare_layout=args.compare_layout,
            batch_sizes=batch_sizes,
        )
    )


if __name__ == "__main__":
    main()
