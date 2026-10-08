"""WoE binning, monotonic pooling, and scorecard points."""

import math

import numpy as np
import pandas as pd
import pytest

from src.scorecard import (
    BASE_SCORE,
    PDO,
    fit_feature_bins,
    pool_monotone_bins,
    score_from_logit,
    woe_from_counts,
    ScorecardModel,
)


def test_woe_matches_smoothed_log_ratio():
    n_bins = 2
    goods = (10 + 0.5) / (40 + 0.5 * n_bins)
    bads = (2 + 0.5) / (10 + 0.5 * n_bins)
    expected = math.log(goods / bads)
    assert woe_from_counts(10, 2, 40, 10, n_bins) == pytest.approx(expected)


def test_score_rises_20_points_when_good_odds_double():
    assert score_from_logit(0.0) == pytest.approx(BASE_SCORE)
    # Logit of default falls by ln(2) when odds of non-default double.
    assert score_from_logit(-math.log(2.0)) == pytest.approx(BASE_SCORE + PDO)


def test_pool_enforces_nonincreasing_woe_for_higher_risk():
    bins = [
        {"lo": 0, "hi": 1, "n_good": 10, "n_bad": 10, "members": [0]},
        {"lo": 1, "hi": 2, "n_good": 30, "n_bad": 5, "members": [1]},
        {"lo": 2, "hi": 3, "n_good": 8, "n_bad": 20, "members": [2]},
    ]
    pooled = pool_monotone_bins(bins, monotone=1, total_good=48, total_bad=35)
    woes = [
        woe_from_counts(item["n_good"], item["n_bad"], 48, 35, len(pooled))
        for item in pooled
    ]
    assert woes == sorted(woes, reverse=True)
    assert len(pooled) < 3


def test_points_reconstruct_the_logit_score():
    rng = np.random.default_rng(0)
    n = 800
    x1 = rng.integers(0, 5, n).astype(float)
    x2 = rng.normal(size=n)
    logits = -1.2 + 0.45 * x1 + 0.35 * x2
    probabilities = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probabilities)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "label": y})
    train = frame.iloc[:500]
    val = frame.iloc[500:]
    model = ScorecardModel(monotone={"x1": 1, "x2": 1}, max_bins=5, min_count=20)
    model.fit(train[["x1", "x2"]], train["label"], val[["x1", "x2"]], val["label"])
    points = model.point_scores(val[["x1", "x2"]])
    expected = BASE_SCORE - (PDO / math.log(2)) * model.predict_logit(val[["x1", "x2"]])
    assert points == pytest.approx(expected)
    table = model.points_table()
    assert set(table["feature"]) >= {"__INTERCEPT__", "x1", "x2"}
    spec = model.bins_["x1"]
    assert spec.woe == sorted(spec.woe, reverse=True)


def test_unseen_discrete_level_maps_to_neutral_woe():
    values = np.array([0, 0, 0, 1, 1, 1, 1], dtype=float)
    y = np.array([0, 0, 1, 1, 1, 1, 0])
    spec = fit_feature_bins(values, y, feature="pay", monotone=1, max_bins=4, min_count=1)
    from src.scorecard import transform_feature

    transformed = transform_feature(np.array([0.0, 1.0, 9.0]), spec)
    assert transformed[2] == pytest.approx(0.0)
    assert transformed[0] != pytest.approx(transformed[1])
