import json
import os
from pathlib import Path

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "4")


def pytest_configure(config):
    Path("results").mkdir(exist_ok=True)


def pytest_unconfigure(config):
    """After pytest-cov writes its JSON report, record that percentage in metrics.json."""
    coverage_path = Path("results/coverage.json")
    metrics_path = Path("results/metrics.json")
    if not coverage_path.exists() or not metrics_path.exists():
        return
    payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    percent = float(payload["totals"]["percent_covered"])
    from src.reporting import apply_coverage, update_readme, write_json

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    updated = apply_coverage(metrics, percent)
    write_json(metrics_path, updated)
    readme = Path("README.md")
    if readme.exists() and "<!-- RESULTS_START -->" in readme.read_text(encoding="utf-8"):
        if updated.get("project") == "credit-risk-scorecard" and "runtime_seconds" in updated:
            update_readme(readme, updated)
