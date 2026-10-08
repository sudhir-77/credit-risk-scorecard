"""Metric functions against hand-computed values."""

import math

import numpy as np
import pytest

from src.metrics import (
    auroc,
    brier_score,
    classification_metrics,
    expected_calibration_error,
    gini_from_auroc,
    ks_cutoff,
    ks_statistic,
)


Y = np.array([0, 1, 0, 1])
SCORES = np.array([0.2, 0.4, 0.6, 0.9])


def test_auroc_and_gini_hand_computed():
    # Pairs of (non-default, default): 3 of 4 are ranked correctly.
    assert auroc(Y, SCORES) == pytest.approx(0.75)
    assert gini_from_auroc(0.75) == pytest.approx(0.5)


def test_brier_hand_computed():
    expected = (0.2**2 + 0.6**2 + 0.6**2 + 0.1**2) / 4
    assert brier_score(Y, SCORES) == pytest.approx(expected)
    assert expected == pytest.approx(0.1925)


def test_ks_hand_computed():
    # Sorted scores 0.2, 0.4, 0.6, 0.9 with labels 0, 1, 0, 1.
    # CDF gaps are 0.5, 0.0, 0.5, 0.0.
    assert ks_statistic(Y, SCORES) == pytest.approx(0.5)


def test_ece_two_bins_hand_computed():
    # Bin [0, 0.5): scores 0.1 and 0.2, labels 0, 0.
    # Bin [0.5, 1]: scores 0.8 and 0.9, labels 1, 1.
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.15)


def test_ece_perfect_predictions_are_zero():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0)


def test_ks_cutoff_separates_classes():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    cutoff = ks_cutoff(y, scores)
    assert 0.3 <= cutoff <= 0.7
    declined = scores >= cutoff
    assert declined.sum() == 3
    assert y[declined].sum() == 3


def test_classification_metrics_bundle():
    bundle = classification_metrics(Y, SCORES)
    assert set(bundle) == {"auroc", "ks", "gini", "brier", "ece"}
    assert bundle["gini"] == pytest.approx(2 * bundle["auroc"] - 1)


def test_ks_requires_both_classes():
    with pytest.raises(ValueError):
        ks_statistic(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))


def test_score_points_identity_matches_logit_formula():
    # Kept here so the PDO factor used by the scorecard stays tied to a known number.
    assert math.isclose(20 / math.log(2), 28.85390081777927, rel_tol=1e-12)
