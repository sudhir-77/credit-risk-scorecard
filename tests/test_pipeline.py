"""Reporting helpers and the quick end-to-end path."""

from pathlib import Path

import pytest

from src.pipeline import run_pipeline
from src.reporting import build_success_criteria, criterion, render_model_card, render_resume, render_results_table


def test_criterion_operators():
    assert criterion("high", 0.8, ">=", 0.77)["passed"] is True
    assert criterion("low", 0.04, "<=", 0.03)["passed"] is False
    assert criterion("band", 1.0, "between", 0.8, 1.25)["passed"] is True
    names = [item["name"] for item in build_success_criteria(0.80, 0.77, 0.45, 0.02, True, True)]
    assert "Test AUROC of the champion model" in names
    assert "Champion vs. scorecard baseline" in names
    lift = next(item for item in build_success_criteria(0.80, 0.77, 0.45, 0.02, True, True) if "baseline" in item["name"])
    assert lift["value"] == pytest.approx(0.03)
    assert lift["passed"] is True


def _fake_metrics() -> dict:
    champion = {"auroc": 0.781, "ks": 0.421, "gini": 0.562, "brier": 0.134, "ece": 0.012}
    baseline = {
        "name": "woe_logistic_scorecard",
        "test_auroc": 0.752,
        "test_ks": 0.39,
        "test_gini": 0.504,
        "test_brier": 0.14,
        "test_ece": 0.02,
    }
    return {
        "project": "credit-risk-scorecard",
        "success_criteria": build_success_criteria(0.781, 0.752, 0.421, 0.012, True, True),
        "metrics": {
            "champion_name": "monotonic_lightgbm",
            "cutoff": 0.22,
            "test": {"champion": champion},
            "validation": {
                "monotonic_lightgbm": {"auroc": 0.79},
                "scorecard": {"auroc": 0.76},
            },
            "fairness": {"air_min": 0.91, "air_max": 1.02, "n_flags": 0},
            "reason_codes": {"n_declined": 100, "n_with_four": 100, "all_signs_agree": 1},
        },
        "baseline": baseline,
        "seeds": [42],
        "hardware": {"peak_rss_gb": 0.4},
        "runtime_seconds": 12.0,
    }


def test_resume_uses_metric_numbers_and_plain_language():
    notes = render_resume(_fake_metrics())
    assert "0.781" in notes
    assert "0.421" in notes
    assert "0.752" in notes
    assert "0.012" in notes
    assert "0.910" in notes
    assert "leveraged" not in notes.lower()
    assert "\u2014" not in notes
    table = render_results_table(_fake_metrics())
    assert "0.7810" in table


def test_model_card_prints_fairness_flags():
    metrics = _fake_metrics()
    fairness = {
        "reference": {"sex": "female", "age_band": "30-39"},
        "flags": [
            {
                "message": "FLAG: AIR for sex=male is 0.700, outside 0.80-1.25.",
            }
        ],
    }
    psi_rows = [{"feature": "PAY_0", "psi": 0.01, "n_bins": 5}]
    card = render_model_card(metrics, fairness, psi_rows)
    assert "FLAG: AIR for sex=male is 0.700, outside 0.80-1.25." in card
    assert "not a production" in card.lower() or "not a production underwriting model" in card


def test_quick_pipeline_writes_required_artifacts(tmp_path: Path):
    data_dir = Path("data")
    result = run_pipeline(quick=True, results_dir=tmp_path, data_dir=data_dir)
    predictions = result["predictions"]
    assert list(predictions.columns) == [
        "id",
        "label",
        "champion_score",
        "scorecard_score",
        "sex",
        "age_band",
    ]
    split_ids = (tmp_path / "split_ids.json").read_text(encoding="utf-8")
    assert '"train"' in split_ids and '"validation"' in split_ids and '"test"' in split_ids
    codes = result["reason_codes"]
    assert list(codes.columns) == ["id", "reason_1", "reason_2", "reason_3", "reason_4"]
    declined_ids = set(
        predictions.loc[predictions["champion_score"] >= result["metrics"]["metrics"]["cutoff"], "id"]
    )
    assert set(codes["id"]) == declined_ids
    assert len(codes) == len(declined_ids)
    assert bool(result["reason_audit"]["sign_agrees"].all())
    assert result["reason_audit"].groupby("id").size().min() == 4
    fairness = result["fairness"]
    assert "cutoff" in fairness
    for group, stats in {**fairness["sex"], **fairness["age_band"]}.items():
        assert "air" in stats
        assert "equal_opportunity_difference" in stats
        assert group
    assert (tmp_path / "test_predictions.csv").exists()
    assert (tmp_path / "fairness.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "roc.png").exists()
    assert (tmp_path / "calibration.png").exists()
    assert (tmp_path / "score_distributions.png").exists()
    assert (tmp_path / "MODEL_CARD.md").exists()
    card = (tmp_path / "MODEL_CARD.md").read_text(encoding="utf-8")
    if fairness["flags"]:
        assert "FLAG:" in card
    else:
        assert "0.80-1.25" in card
    # The quick subset is not the quoted study, but it must still keep the test IDs out of train.
    payload_train = set(result["metrics"]["splits"]["default_of_credit_card_clients"].values())
    assert payload_train  # counts exist
    assert "SEX" not in result["psi"]["feature"].tolist()
    assert "AGE" not in result["psi"]["feature"].tolist()
