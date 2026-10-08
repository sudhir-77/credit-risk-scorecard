"""Discrimination and calibration metrics.

Scores are probabilities (or any ranking) where a larger value means a higher
chance of the positive class, default next month.
"""

from __future__ import annotations

import numpy as np

from sklearn.metrics import roc_auc_score


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(np.asarray(y_true), np.asarray(scores)))


def gini_from_auroc(auc: float) -> float:
    """Credit Gini, equal to 2 * AUROC - 1."""
    return float(2.0 * auc - 1.0)


def brier_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(scores, dtype=float)
    return float(np.mean((p - y) ** 2))


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between score CDFs of the two classes."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(s, kind="mergesort")
    y_sorted = y[order]
    n_pos = int(y_sorted.sum())
    n_neg = int(len(y_sorted) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("KS requires both classes")
    cdf_pos = np.cumsum(y_sorted) / n_pos
    cdf_neg = np.cumsum(1 - y_sorted) / n_neg
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def expected_calibration_error(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Equal-width ECE. Empty bins are skipped."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    if len(y) == 0:
        raise ValueError("ECE requires at least one row")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(len(y))
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(float(y[mask].mean()) - float(p[mask].mean()))
        ece += (count / total) * gap
    return float(ece)


def classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    auc = auroc(y_true, scores)
    return {
        "auroc": auc,
        "ks": ks_statistic(y_true, scores),
        "gini": gini_from_auroc(auc),
        "brier": brier_score(y_true, scores),
        "ece": expected_calibration_error(y_true, scores, n_bins=10),
    }


def ks_cutoff(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Validation cutoff: decline when score >= threshold that maximizes TPR - FPR."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    thresholds = np.unique(s)
    if len(thresholds) > 201:
        thresholds = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 199)))
    best_t = float(np.median(s))
    best_j = -np.inf
    for threshold in thresholds:
        predicted = s >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fn = int(np.sum(~predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        tn = int(np.sum(~predicted & (y == 0)))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        youden = tpr - fpr
        if youden > best_j:
            best_j = youden
            best_t = float(threshold)
    return best_t
