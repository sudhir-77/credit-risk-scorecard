"""Credit default scorecard: WoE logistic baseline vs monotonic LightGBM."""

from __future__ import annotations

import os

# Cap BLAS threads before numpy is imported by submodules.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "4")

__version__ = "1.0.0"
