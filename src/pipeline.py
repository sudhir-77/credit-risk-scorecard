"""End-to-end train, evaluate, and write artifacts. The test split is scored once."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (
    MODEL_FEATURES,
    MONOTONE_CONSTRAINTS,
    SEED,
    assert_split_integrity,
    feature_matrix,
    load_modeling_frame,
    make_splits,
    project_root,
    split_id_payload,
)
from src.explain import build_reason_tables, training_references
from src.fairness import fairness_is_complete, fairness_report
from src.gbm import IsotonicCalibrator, MonotonicGBM
from src.metrics import classification_metrics, ks_cutoff
from src.plots import save_evaluation_plots
from src.reporting import (
    build_success_criteria,
    render_model_card,
    render_resume,
    update_readme,
    write_json,
)
from src.scorecard import ScorecardModel
from src.stability import psi_table


def _hardware() -> dict:
    import platform

    import psutil

    process = psutil.Process()
    info = process.memory_info()
    peak = getattr(info, "peak_wset", info.rss)
    cpu = platform.processor() or platform.machine()
    return {
        "cpu": cpu,
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": None,
        "python": platform.python_version(),
        "os": platform.platform(),
        "peak_rss_gb": round(float(peak) / (1024**3), 3),
    }


def run_pipeline(
    quick: bool = False,
    results_dir: Path | None = None,
    data_dir: Path | None = None,
    root: Path | None = None,
) -> dict:
    started = time.perf_counter()
    root = root or project_root()
    data_dir = data_dir or (root / "data")
    results_dir = results_dir or (root / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    frame = load_modeling_frame(data_dir, quick=quick, seed=SEED)
    splits = make_splits(frame, seed=SEED)
    id_payload = split_id_payload(splits)
    for key in id_payload:
        id_payload[key] = sorted(id_payload[key])
    assert_split_integrity(id_payload)
    write_json(results_dir / "split_ids.json", id_payload)

    x_train = feature_matrix(splits["train"])
    x_val = feature_matrix(splits["validation"])
    x_test = feature_matrix(splits["test"])
    y_train = splits["train"]["label"].to_numpy()
    y_val = splits["validation"]["label"].to_numpy()
    y_test = splits["test"]["label"].to_numpy()

    scorecard = ScorecardModel(
        monotone=MONOTONE_CONSTRAINTS,
        max_bins=6 if quick else 8,
        min_count=8 if quick else 50,
    )
    scorecard.fit(x_train, y_train, x_val, y_val)
    gbm = MonotonicGBM(quick=quick, seed=SEED)
    gbm.fit(x_train, y_train, x_val, y_val)

    val_gbm_raw = gbm.predict_proba(x_val)
    val_scorecard_raw = scorecard.predict_proba(x_val)
    gbm_calibrator = IsotonicCalibrator().fit(val_gbm_raw, y_val)
    scorecard_calibrator = IsotonicCalibrator().fit(val_scorecard_raw, y_val)
    # Six decimals round-trip through CSV, so the decline rule matches the saved file.
    val_gbm = _round_prob(gbm_calibrator.transform(val_gbm_raw))
    val_scorecard = _round_prob(scorecard_calibrator.transform(val_scorecard_raw))
    val_gbm_auc = float(classification_metrics(y_val, val_gbm)["auroc"])
    val_scorecard_auc = float(classification_metrics(y_val, val_scorecard)["auroc"])
    champion_name = (
        "monotonic_lightgbm" if val_gbm_auc >= val_scorecard_auc else "woe_logistic_scorecard"
    )
    val_champion = val_gbm if champion_name == "monotonic_lightgbm" else val_scorecard
    cutoff = float(np.round(ks_cutoff(y_val, val_champion), 6))

    # Test labels are not used above this line.
    test_gbm = _round_prob(gbm_calibrator.transform(gbm.predict_proba(x_test)))
    test_scorecard = _round_prob(scorecard_calibrator.transform(scorecard.predict_proba(x_test)))
    test_champion = test_gbm if champion_name == "monotonic_lightgbm" else test_scorecard

    predictions = pd.DataFrame(
        {
            "id": splits["test"]["ID"].astype(int),
            "label": y_test.astype(int),
            "champion_score": test_champion,
            "scorecard_score": test_scorecard,
            "gbm_score": test_gbm,
            "sex": splits["test"]["sex"].astype(str),
            "age_band": splits["test"]["age_band"].astype(str),
        }
    ).sort_values("id").reset_index(drop=True)
    export_columns = [
        "id",
        "label",
        "champion_score",
        "scorecard_score",
        "sex",
        "age_band",
    ]
    predictions.loc[:, export_columns].to_csv(results_dir / "test_predictions.csv", index=False)
    reloaded = pd.read_csv(results_dir / "test_predictions.csv")
    if not np.array_equal(reloaded["champion_score"].to_numpy(), predictions["champion_score"].to_numpy()):
        raise RuntimeError("Champion scores did not round-trip through test_predictions.csv")
    if not np.array_equal(reloaded["scorecard_score"].to_numpy(), predictions["scorecard_score"].to_numpy()):
        raise RuntimeError("Scorecard scores did not round-trip through test_predictions.csv")

    declined = predictions["champion_score"].to_numpy() >= cutoff
    ordered_test = splits["test"].set_index("ID").loc[predictions["id"]]
    shap_frame = feature_matrix(ordered_test.reset_index())
    shap_values = gbm.shap_values(shap_frame)
    references = training_references(x_train, MONOTONE_CONSTRAINTS)
    codes, audit = build_reason_tables(
        predictions["id"].to_numpy(),
        shap_frame,
        shap_values,
        declined,
        MONOTONE_CONSTRAINTS,
        references,
    )
    codes.to_csv(results_dir / "reason_codes.csv", index=False)
    audit.to_csv(results_dir / "reason_code_audit.csv", index=False)

    fairness = fairness_report(
        predictions["label"].to_numpy(),
        predictions["champion_score"].to_numpy(),
        predictions["sex"],
        predictions["age_band"],
        cutoff,
    )
    write_json(results_dir / "fairness.json", fairness)
    saved_fairness = json.loads((results_dir / "fairness.json").read_text(encoding="utf-8"))
    saved_cutoff = float(saved_fairness["cutoff"])
    saved_declined = set(
        reloaded.loc[reloaded["champion_score"] >= saved_cutoff, "id"].astype(int)
    )
    if saved_declined != set(int(value) for value in predictions.loc[declined, "id"]):
        raise RuntimeError("Declined IDs changed when scores and the cutoff were reloaded")

    psi = psi_table(x_train, x_test, n_bins=10)
    psi.to_csv(results_dir / "psi.csv", index=False)
    scorecard.points_table().to_csv(results_dir / "scorecard_points.csv", index=False)
    save_evaluation_plots(
        predictions["label"].to_numpy(),
        predictions["champion_score"].to_numpy(),
        predictions["scorecard_score"].to_numpy(),
        results_dir,
    )

    # Metrics use the sorted file columns so each score stays paired with its label.
    champion_metrics = classification_metrics(
        predictions["label"].to_numpy(),
        predictions["champion_score"].to_numpy(),
    )
    gbm_metrics = classification_metrics(
        predictions["label"].to_numpy(),
        predictions["gbm_score"].to_numpy(),
    )
    scorecard_metrics = classification_metrics(
        predictions["label"].to_numpy(),
        predictions["scorecard_score"].to_numpy(),
    )
    predictions = predictions.drop(columns=["gbm_score"])
    reason_ok = _reason_codes_ok(codes, audit, int(declined.sum()))
    fairness_ok = fairness_is_complete(fairness)
    airs = [
        float(stats["air"])
        for attribute in ("sex", "age_band")
        for stats in fairness[attribute].values()
    ]
    runtime = time.perf_counter() - started
    metrics = {
        "project": "credit-risk-scorecard",
        "success_criteria": build_success_criteria(
            champion_metrics["auroc"],
            scorecard_metrics["auroc"],
            champion_metrics["ks"],
            champion_metrics["ece"],
            reason_ok,
            fairness_ok,
        ),
        "metrics": {
            "champion_name": champion_name,
            "cutoff": float(cutoff),
            "n_rows": int(len(frame)),
            "n_features": int(len(MODEL_FEATURES)),
            "quick": bool(quick),
            "test": {
                "champion": champion_metrics,
                "monotonic_lightgbm": gbm_metrics,
                "scorecard": scorecard_metrics,
            },
            "validation": {
                "monotonic_lightgbm": {"auroc": val_gbm_auc},
                "scorecard": {"auroc": val_scorecard_auc},
            },
            "fairness": {
                "air_min": float(min(airs)),
                "air_max": float(max(airs)),
                "n_flags": int(len(fairness["flags"])),
            },
            "psi": {
                "max": float(psi["psi"].max()),
                "mean": float(psi["psi"].mean()),
            },
            "reason_codes": {
                "n_declined": int(declined.sum()),
                "n_with_four": int(len(codes)),
                "all_signs_agree": 1.0 if reason_ok else 0.0,
                "share_positive_shap": float((audit["shap"] > 0).mean()) if len(audit) else 0.0,
            },
            "gbm_best_iteration": int(gbm.best_iteration_ or 0),
            "scorecard_C": float(scorecard.C_ or 0.0),
        },
        "baseline": {
            "name": "woe_logistic_scorecard",
            "test_auroc": scorecard_metrics["auroc"],
            "test_ks": scorecard_metrics["ks"],
            "test_gini": scorecard_metrics["gini"],
            "test_brier": scorecard_metrics["brier"],
            "test_ece": scorecard_metrics["ece"],
        },
        "splits": {
            "default_of_credit_card_clients": {
                "train": int(len(splits["train"])),
                "val": int(len(splits["validation"])),
                "test": int(len(splits["test"])),
            }
        },
        "seeds": [SEED],
        "hardware": _hardware(),
        "runtime_seconds": float(runtime),
    }
    write_json(results_dir / "metrics.json", metrics)
    card = render_model_card(metrics, fairness, psi.to_dict(orient="records"))
    (results_dir / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    if results_dir.resolve() == (root / "results").resolve():
        (root / "MODEL_CARD.md").write_text(card, encoding="utf-8")
        (root / "RESUME_NOTES.md").write_text(render_resume(metrics), encoding="utf-8")
        readme = root / "README.md"
        if readme.exists():
            update_readme(readme, metrics)
    return {
        "metrics": metrics,
        "fairness": fairness,
        "predictions": predictions,
        "reason_codes": codes,
        "reason_audit": audit,
        "psi": psi,
    }


def _round_prob(scores: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(scores, dtype=float), 6)


def _reason_codes_ok(codes: pd.DataFrame, audit: pd.DataFrame, n_declined: int) -> bool:
    if n_declined <= 0 or len(codes) != n_declined or audit.empty:
        return False
    required = ["reason_1", "reason_2", "reason_3", "reason_4"]
    if codes[required].isna().any().any():
        return False
    if codes[required].eq("").any().any():
        return False
    counts = audit.groupby("id").size()
    if int(counts.min()) != 4 or int(counts.max()) != 4:
        return False
    if not bool(audit["sign_agrees"].all()):
        return False
    return set(codes["id"]).issubset(set(audit["id"]))
