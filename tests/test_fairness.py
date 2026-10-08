"""Hand-computed AIR, equal-opportunity difference, and PSI."""

import math

import numpy as np
import pandas as pd
import pytest

from src.fairness import fairness_is_complete, fairness_report
from src.stability import feature_psi, psi_from_proportions


def test_air_and_equal_opportunity_hand_computed():
    # 4 female (largest group) and 2 male.
    # Female approvals: scores 0.1, 0.2, 0.6, 0.7 against cutoff 0.5 -> 2/4 = 0.5.
    # Male approvals: 0.1 and 0.8 -> 1/2 = 0.5. AIR = 1.
    # Goods (label 0): female indexes 0, 1, 2 -> approvals T, T, F -> TPR 2/3.
    # Male goods: index 4 only -> approval T -> TPR 1. EOD = 1 - 2/3.
    y = np.array([0, 0, 0, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.7, 0.1, 0.8])
    sex = pd.Series(["female", "female", "female", "female", "male", "male"])
    age = pd.Series(["<30", "30-39", "40+", "<30", "30-39", "40+"])
    report = fairness_report(y, scores, sex, age, cutoff=0.5)
    assert report["cutoff"] == pytest.approx(0.5)
    assert report["reference"]["sex"] == "female"
    assert report["sex"]["female"]["approval_rate"] == pytest.approx(0.5)
    assert report["sex"]["male"]["approval_rate"] == pytest.approx(0.5)
    assert report["sex"]["male"]["air"] == pytest.approx(1.0)
    assert report["sex"]["female"]["tpr_good"] == pytest.approx(2 / 3)
    assert report["sex"]["male"]["tpr_good"] == pytest.approx(1.0)
    assert report["sex"]["male"]["equal_opportunity_difference"] == pytest.approx(1 - 2 / 3)
    assert fairness_is_complete(report)


def test_air_outside_band_is_flagged():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1])
    sex = pd.Series(["female"] * 5 + ["male"] * 4)
    age = pd.Series(["<30", "30-39", "40+", "<30", "30-39", "40+", "<30", "30-39", "40+"])
    report = fairness_report(y, scores, sex, age, cutoff=0.5)
    # Female approval 5/5, male approval 1/4, AIR = 0.25.
    assert report["sex"]["male"]["air"] == pytest.approx(0.25)
    messages = [item["message"] for item in report["flags"]]
    assert any(message.startswith("FLAG:") and "sex=male" in message for message in messages)
    assert any("0.80-1.25" in message for message in messages)


def test_psi_hand_computed_proportions():
    expected = np.array([3.0, 2.0])
    actual = np.array([1.0, 4.0])
    # Shares 0.6/0.4 vs 0.2/0.8.
    manual = (0.2 - 0.6) * math.log(0.2 / 0.6) + (0.8 - 0.4) * math.log(0.8 / 0.4)
    assert psi_from_proportions(expected, actual) == pytest.approx(manual)
    assert manual == pytest.approx(0.7167037877)


def test_psi_edges_come_from_train_distribution():
    train = np.array([0.0, 0.0, 0.0, 1.0])
    test = np.array([5.0, 5.0, 5.0, 5.0])
    psi, n_bins = feature_psi(train, test, n_bins=10)
    assert n_bins >= 1
    assert psi > 0
