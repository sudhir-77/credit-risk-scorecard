"""Reproduce the credit-default study from scratch.

Usage:
    python run.py
    python run.py --quick
    python run.py --device cpu
"""

from __future__ import annotations

import argparse
import os

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "4")

from pathlib import Path

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit default scorecard vs monotonic LightGBM")
    parser.add_argument("--quick", action="store_true", help="Run a small stratified subset")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Accepted for compatibility. Training stays on CPU.",
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.device == "cuda":
        print("CUDA was requested. This project trains LightGBM and scikit-learn on CPU.")
    result = run_pipeline(quick=args.quick, results_dir=args.results_dir)
    metrics = result["metrics"]
    champion = metrics["metrics"]["test"]["champion"]
    print(f"champion={metrics['metrics']['champion_name']}")
    print(
        "test "
        f"AUROC={champion['auroc']:.4f} "
        f"KS={champion['ks']:.4f} "
        f"ECE={champion['ece']:.4f} "
        f"runtime_s={metrics['runtime_seconds']:.1f} "
        f"peak_rss_gb={metrics['hardware']['peak_rss_gb']:.3f}"
    )
    for item in metrics["success_criteria"]:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status} {item['name']}: {item['value']:.4f} {item['op']} {item['threshold']}")


if __name__ == "__main__":
    main()
