"""Population stability index for model inputs, train versus test."""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi_from_proportions(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI = sum((actual - expected) * ln(actual / expected)), with empty bins clipped."""
    expected_arr = np.asarray(expected, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    if expected_arr.shape != actual_arr.shape:
        raise ValueError("Proportion vectors must match")
    if expected_arr.sum() <= 0 or actual_arr.sum() <= 0:
        raise ValueError("Proportion vectors must have positive mass")
    expected_share = np.clip(expected_arr / expected_arr.sum(), 1e-6, None)
    actual_share = np.clip(actual_arr / actual_arr.sum(), 1e-6, None)
    expected_share = expected_share / expected_share.sum()
    actual_share = actual_share / actual_share.sum()
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def _bin_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indexes = np.digitize(values, edges[1:-1], right=False)
    return np.bincount(indexes, minlength=len(edges) - 1).astype(float)


def feature_psi(train: np.ndarray, test: np.ndarray, n_bins: int = 10) -> tuple[float, int]:
    """PSI using quantile edges fit on train only."""
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    edges = np.unique(np.quantile(train, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        levels = np.unique(train)
        # Discrete fallback: one bin per training level, unseen test levels join the last bin.
        train_counts = np.array([(train == level).sum() for level in levels], dtype=float)
        test_counts = np.array([(test == level).sum() for level in levels], dtype=float)
        return psi_from_proportions(train_counts, test_counts), int(len(levels))
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    train_counts = _bin_counts(train, edges)
    test_counts = _bin_counts(test, edges)
    return psi_from_proportions(train_counts, test_counts), int(len(edges) - 1)


def psi_table(train: pd.DataFrame, test: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for column in train.columns:
        psi, bins = feature_psi(train[column].to_numpy(), test[column].to_numpy(), n_bins=n_bins)
        rows.append({"feature": column, "psi": psi, "n_bins": bins})
    return pd.DataFrame(rows)
