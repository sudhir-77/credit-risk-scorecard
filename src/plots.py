"""ROC, calibration, and score-distribution plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def save_evaluation_plots(
    y_true: np.ndarray,
    champion: np.ndarray,
    scorecard: np.ndarray,
    directory: Path,
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y_true, dtype=int)
    champion = np.asarray(champion, dtype=float)
    scorecard = np.asarray(scorecard, dtype=float)
    written = [
        _roc(y, champion, scorecard, directory / "roc.png"),
        _calibration(y, champion, scorecard, directory / "calibration.png"),
        _distributions(y, champion, directory / "score_distributions.png"),
    ]
    return [path.name for path in written]


def _roc(y: np.ndarray, champion: np.ndarray, scorecard: np.ndarray, path: Path) -> Path:
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    for scores, label in ((champion, "Champion"), (scorecard, "WoE scorecard")):
        fpr, tpr, _ = roc_curve(y, scores)
        axis.plot(fpr, tpr, label=label)
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC on the held-out test set")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _calibration(y: np.ndarray, champion: np.ndarray, scorecard: np.ndarray, path: Path) -> Path:
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    for scores, label in ((champion, "Champion"), (scorecard, "WoE scorecard")):
        centers, rates = _reliability(y, scores, n_bins=10)
        axis.plot(centers, rates, marker="o", label=label)
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Perfect")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed default rate")
    axis.set_title("Calibration on the held-out test set")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _reliability(y: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> tuple[list[float], list[float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers: list[float] = []
    rates: list[float] = []
    clipped = np.clip(scores, 0.0, 1.0)
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (clipped >= lower) & (clipped <= upper)
        else:
            mask = (clipped >= lower) & (clipped < upper)
        if int(mask.sum()) == 0:
            continue
        centers.append(float(clipped[mask].mean()))
        rates.append(float(y[mask].mean()))
    return centers, rates


def _distributions(y: np.ndarray, champion: np.ndarray, path: Path) -> Path:
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.hist(champion[y == 0], bins=30, alpha=0.65, density=True, label="No default")
    axis.hist(champion[y == 1], bins=30, alpha=0.65, density=True, label="Default")
    axis.set_xlabel("Calibrated probability of default")
    axis.set_ylabel("Density")
    axis.set_title("Champion score distribution on the test set")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
