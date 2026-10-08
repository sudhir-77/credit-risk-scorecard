"""Reason-code direction must match the monotonic constraint."""

import numpy as np
import pandas as pd
import pytest

from src.explain import (
    build_reason_tables,
    reason_code,
    select_reason_codes,
    sign_agrees,
    training_references,
)
from src.gbm import _fit_one


def test_sign_agreement_rules():
    assert sign_agrees("PAY_0_HIGH", 1, value=2, reference=0)
    assert not sign_agrees("PAY_0_HIGH", 1, value=0, reference=0)
    assert sign_agrees("LIMIT_BAL_LOW", -1, value=100, reference=500)
    assert not sign_agrees("LIMIT_BAL_HIGH", -1, value=100, reference=500)
    assert reason_code("PAY_AMT1", -1) == "PAY_AMT1_LOW"


def test_selector_keeps_four_adverse_codes_and_ranks_by_shap():
    columns = ["PAY_0", "PAY_2", "LIMIT_BAL", "PAY_AMT1", "EDUCATION"]
    constraints = {"PAY_0": 1, "PAY_2": 1, "LIMIT_BAL": -1, "PAY_AMT1": -1, "EDUCATION": 0}
    references = {
        "PAY_0": {"median": 0, "favorable_quartile": -1, "favorable_decile": -2, "favorable_extreme": -2},
        "PAY_2": {"median": 0, "favorable_quartile": -1, "favorable_decile": -2, "favorable_extreme": -2},
        "LIMIT_BAL": {"median": 500, "favorable_quartile": 800, "favorable_decile": 1000, "favorable_extreme": 2000},
        "PAY_AMT1": {"median": 100, "favorable_quartile": 200, "favorable_decile": 400, "favorable_extreme": 900},
        "EDUCATION": {"median": 2, "favorable_quartile": 2, "favorable_decile": 2, "favorable_extreme": 2},
    }
    values = np.array([2.0, 1.0, 100.0, 10.0, 3.0])
    shap_row = np.array([0.5, 0.2, 0.4, 0.3, 0.9])
    chosen = select_reason_codes(values, shap_row, columns, constraints, references, k=4)
    assert [item["code"] for item in chosen] == [
        "PAY_0_HIGH",
        "LIMIT_BAL_LOW",
        "PAY_AMT1_LOW",
        "PAY_2_HIGH",
    ]
    assert all(item["sign_agrees"] for item in chosen)
    assert all(item["shap"] > 0 for item in chosen)
    # EDUCATION has the largest SHAP but no constraint, so it is not a reason.
    assert "EDUCATION" not in {item["feature"] for item in chosen}


def test_positive_shap_on_the_favorable_side_is_not_used():
    columns = ["PAY_0"]
    constraints = {"PAY_0": 1}
    references = {
        "PAY_0": {"median": 0, "favorable_quartile": 0, "favorable_decile": 0, "favorable_extreme": 0}
    }
    # Value is not above any reference, so a positive SHAP still disagrees.
    with pytest.raises(RuntimeError):
        select_reason_codes(
            np.array([-1.0]),
            np.array([0.8]),
            columns,
            constraints,
            references,
            k=1,
        )


def test_tables_cover_every_declined_id():
    frame = pd.DataFrame(
        {
            "PAY_0": [2.0, 0.0, 3.0],
            "LIMIT_BAL": [50.0, 900.0, 40.0],
            "PAY_AMT1": [5.0, 400.0, 1.0],
            "PAY_AMT2": [5.0, 400.0, 1.0],
        }
    )
    constraints = {"PAY_0": 1, "LIMIT_BAL": -1, "PAY_AMT1": -1, "PAY_AMT2": -1}
    references = training_references(
        pd.DataFrame(
            {
                "PAY_0": [-2, -1, 0, 1, 2],
                "LIMIT_BAL": [100, 200, 400, 800, 1600],
                "PAY_AMT1": [0, 10, 50, 200, 800],
                "PAY_AMT2": [0, 10, 50, 200, 800],
            }
        ),
        constraints,
    )
    shap_values = np.array(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.4, 0.3, 0.2],
        ]
    )
    declined = np.array([True, False, True])
    codes, audit = build_reason_tables(
        np.array([10, 11, 12]),
        frame,
        shap_values,
        declined,
        constraints,
        references,
    )
    assert codes["id"].tolist() == [10, 12]
    assert audit.groupby("id").size().tolist() == [4, 4]
    assert bool(audit["sign_agrees"].all())


def test_tree_shap_matches_lightgbm_and_respects_monotonicity():
    import shap

    rng = np.random.default_rng(0)
    n = 240
    frame = pd.DataFrame(
        {
            "PAY_0": rng.integers(-1, 4, n).astype(float),
            "PAY_AMT1": rng.uniform(0, 500, n),
        }
    )
    logits = -0.4 + 0.7 * frame["PAY_0"] - 0.004 * frame["PAY_AMT1"]
    probabilities = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probabilities)
    train, val = frame.iloc[:180], frame.iloc[180:]
    y_train, y_val = y[:180], y[180:]
    model = _fit_one(
        train,
        y_train,
        val,
        y_val,
        {
            "max_depth": 2,
            "num_leaves": 4,
            "min_child_samples": 15,
            "learning_rate": 0.1,
            "n_estimators": 25,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
        constraints=[1, -1],
        scale_pos_weight=1.0,
        seed=0,
    )
    booster = model.booster_
    sample = val.iloc[:30]
    margin = booster.predict(sample, raw_score=True)
    contrib = np.asarray(booster.predict(sample, pred_contrib=True))
    assert contrib.shape == (len(sample), 3)
    assert contrib[:, :-1].sum(axis=1) + contrib[:, -1] == pytest.approx(margin, rel=1e-5, abs=1e-5)

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    correlation = np.corrcoef(shap_values.ravel(), contrib[:, :-1].ravel())[0, 1]
    assert correlation > 0.99

    base = sample.copy()
    increased = base.copy()
    increased["PAY_0"] = increased["PAY_0"] + 3
    base_margin = booster.predict(base, raw_score=True)
    increased_margin = booster.predict(increased, raw_score=True)
    assert np.all(increased_margin + 1e-8 >= base_margin)

    decreased_payment = base.copy()
    decreased_payment["PAY_AMT1"] = decreased_payment["PAY_AMT1"] + 400
    paid_more = booster.predict(decreased_payment, raw_score=True)
    assert np.all(paid_more <= base_margin + 1e-8)
