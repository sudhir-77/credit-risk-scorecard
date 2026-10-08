"""metrics.json, model card, resume notes, and the README results table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def criterion(
    name: str,
    value: float,
    op: str,
    threshold: float,
    threshold_high: float | None = None,
) -> dict:
    number = float(value)
    if op == ">=":
        passed = number >= threshold
    elif op == "<=":
        passed = number <= threshold
    elif op == ">":
        passed = number > threshold
    elif op == "<":
        passed = number < threshold
    elif op == "between":
        if threshold_high is None:
            raise ValueError("between requires threshold_high")
        passed = threshold <= number <= threshold_high
    else:
        raise ValueError(f"Unsupported operator {op}")
    return {
        "name": name,
        "value": number,
        "op": op,
        "threshold": float(threshold),
        "threshold_high": None if threshold_high is None else float(threshold_high),
        "passed": bool(passed),
    }


def build_success_criteria(
    champion_auroc: float,
    scorecard_auroc: float,
    champion_ks: float,
    champion_ece: float,
    reason_ok: bool,
    fairness_ok: bool,
) -> list[dict]:
    lift = float(champion_auroc - scorecard_auroc)
    return [
        criterion("Test AUROC of the champion model", champion_auroc, ">=", 0.77),
        criterion("Champion vs. scorecard baseline", lift, ">=", 0.01),
        criterion("KS statistic of the champion", champion_ks, ">=", 0.40),
        criterion("Expected calibration error (10 bins) after calibration", champion_ece, "<=", 0.03),
        criterion("Reason codes", 1.0 if reason_ok else 0.0, ">=", 1.0),
        criterion("Fairness reporting", 1.0 if fairness_ok else 0.0, ">=", 1.0),
    ]


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Refusing to write a non-finite metric")
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")


def render_results_table(metrics: dict) -> str:
    test = metrics["metrics"]["test"]
    champion = test["champion"]
    baseline = metrics["baseline"]
    lines = [
        "| Metric | Champion | WoE scorecard |",
        "|---|---:|---:|",
        f"| AUROC | {champion['auroc']:.4f} | {baseline['test_auroc']:.4f} |",
        f"| KS | {champion['ks']:.4f} | {baseline['test_ks']:.4f} |",
        f"| Gini | {champion['gini']:.4f} | {baseline['test_gini']:.4f} |",
        f"| Brier | {champion['brier']:.4f} | {baseline['test_brier']:.4f} |",
        f"| ECE (10 bins) | {champion['ece']:.4f} | {baseline['test_ece']:.4f} |",
        "",
        (
            f"Champion: `{metrics['metrics']['champion_name']}`."
            f" Cutoff: {metrics['metrics']['cutoff']:.4f}."
            f" Runtime: {metrics['runtime_seconds']:.1f} s."
            f" Peak RSS: {metrics['hardware']['peak_rss_gb']:.3f} GB."
        ),
        "",
        "| Success criterion | Value | Threshold | Passed |",
        "|---|---:|---:|---|",
    ]
    for item in metrics["success_criteria"]:
        threshold = item["threshold"]
        if item["op"] == "between":
            threshold_text = f"{threshold:.4f} to {item['threshold_high']:.4f}"
        else:
            threshold_text = f"{item['op']} {threshold:.4f}"
        lines.append(
            f"| {item['name']} | {item['value']:.4f} | {threshold_text} | {item['passed']} |"
        )
    return "\n".join(lines)


def update_readme(readme: Path, metrics: dict) -> None:
    start = "<!-- RESULTS_START -->"
    end = "<!-- RESULTS_END -->"
    text = readme.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise RuntimeError("README is missing results markers")
    block = f"{start}\n{render_results_table(metrics)}\n{end}"
    pre, rest = text.split(start, 1)
    _, post = rest.split(end, 1)
    readme.write_text(pre + block + post, encoding="utf-8")


def render_resume(metrics: dict) -> str:
    test = metrics["metrics"]["test"]["champion"]
    baseline = metrics["baseline"]
    air_min = metrics["metrics"]["fairness"]["air_min"]
    auroc = f"{test['auroc']:.3f}"
    ks = f"{test['ks']:.3f}"
    base = f"{baseline['test_auroc']:.3f}"
    ece = f"{test['ece']:.3f}"
    air = f"{air_min:.3f}"
    name = metrics["metrics"]["champion_name"]
    champion_label = "monotonic LightGBM" if name == "monotonic_lightgbm" else name
    text = f"""# Resume notes

## Summary

This project compares a weight-of-evidence logistic scorecard with a monotonic LightGBM default model on the UCI Default of Credit Card Clients dataset (30,000 accounts, Taiwan 2005). The champion is chosen on a validation set, then discrimination, calibration, adverse-action reason codes, fairness, and population stability are measured once on a held-out test set.

## Skills and tools

Python, pandas, scikit-learn, LightGBM, SHAP, matplotlib, pytest, logistic regression, gradient boosting, weight of evidence, calibration, fairness metrics

## Draft bullets

- Built a credit-default model on 30,000 real card accounts with {champion_label} at {auroc} AUROC and {ks} KS vs. {base} for a WoE logistic scorecard, calibrated to {ece} expected calibration error
- Generated top-4 adverse-action reason codes from SHAP for every declined test applicant and checked adverse impact ratio across sex and age bands ({air} minimum)
- Documented stability (PSI) and a champion/challenger recommendation in a model-risk write-up

## GitHub description

Monotonic LightGBM vs a WoE credit scorecard with calibration, SHAP reason codes, and fair-lending checks on UCI default data.
"""
    forbidden = ("leveraged", "robust", "cutting-edge", "seamless", "spearheaded", "\u2014")
    lowered = text.lower()
    for word in forbidden:
        if word in lowered or word in text:
            raise RuntimeError(f"Resume notes contain a forbidden phrase: {word}")
    return text


def render_model_card(metrics: dict, fairness: dict, psi_rows: list[dict]) -> str:
    test = metrics["metrics"]["test"]
    champion = test["champion"]
    baseline = metrics["baseline"]
    name = metrics["metrics"]["champion_name"]
    flags = fairness.get("flags", [])
    if flags:
        flag_lines = "\n".join(f"- {item['message']}" for item in flags)
    else:
        flag_lines = "- No group has an adverse impact ratio outside 0.80-1.25 at this cutoff."
    psi_lines = "\n".join(
        f"| {row['feature']} | {row['psi']:.4f} | {row['n_bins']} |" for row in psi_rows
    )
    highest_psi = max(psi_rows, key=lambda row: row["psi"])
    return f"""# Model card: credit default scorecard

This is a portfolio study on a public dataset. It is not a production underwriting model, it was not used to decision real applicants, and it has no regulatory approval.

## Purpose

Compare a weight-of-evidence logistic scorecard with a monotonic LightGBM model on the probability that a credit-card client defaults next month. The write-up records discrimination, calibration, adverse-action reason codes, fair-lending ratios, and population stability.

## Data

- Source: Default of Credit Card Clients, UCI Machine Learning Repository, I-Cheng Yeh and Che-hui Lien.
- License: CC BY 4.0.
- Population: 30,000 Taiwanese credit-card clients from 2005.
- Target: `label` = 1 if the client defaulted the next month.
- Split: stratified 60/20/20 train/validation/test, seed {metrics['seeds'][0]}. Binning, LightGBM hyperparameters, isotonic calibration, and the decline cutoff use train and validation only. The test set is scored once.
- Protected attributes: sex and age are excluded from model inputs and used only in the fairness report.

## Models

- Challenger: weight-of-evidence logistic scorecard. Base score 600, 20 points to double the odds of non-default. Bins are fit on the training set. `C` and class weight are chosen by validation AUROC.
- Candidate: LightGBM with monotone constraints. Repayment status, utilization, and average/max delinquency cannot decrease predicted risk as they rise. Credit limit and payment amounts cannot increase predicted risk as they rise. Education, marriage, and raw bill amounts are unconstrained because their direction is not reliable.
- Both raw scores are passed through isotonic regression fit on validation predictions only.
- Champion selection uses validation AUROC of the calibrated scores. Selected champion: `{name}`.
- Decline rule: calibrated probability of default >= {metrics['metrics']['cutoff']:.6f}. The cutoff maximizes Youden's J (TPR minus FPR) on the validation set.

## Test results

| Metric | Champion | WoE scorecard |
|---|---:|---:|
| AUROC | {champion['auroc']:.4f} | {baseline['test_auroc']:.4f} |
| KS | {champion['ks']:.4f} | {baseline['test_ks']:.4f} |
| Gini | {champion['gini']:.4f} | {baseline['test_gini']:.4f} |
| Brier | {champion['brier']:.4f} | {baseline['test_brier']:.4f} |
| ECE (10 equal-width bins) | {champion['ece']:.4f} | {baseline['test_ece']:.4f} |

Validation AUROC used for selection: monotonic LightGBM {metrics['metrics']['validation']['monotonic_lightgbm']['auroc']:.4f}, scorecard {metrics['metrics']['validation']['scorecard']['auroc']:.4f}.

## Reason codes

For every test applicant whose calibrated probability is at or above the cutoff, the report lists four reason codes. Codes are the monotonic features with the largest TreeSHAP contributions to the default log-odds (LightGBM `pred_contrib`), restricted to features whose value is on the adverse side of a training reference. `PAY_0_HIGH` means a high recent delinquency code, which the constraint treats as higher risk. `PAY_AMT1_LOW` means a low payment, which the constraint treats as higher risk. Declined test applicants: {metrics['metrics']['reason_codes']['n_declined']}.

## Fair lending

Adverse impact ratio is the group's approval rate divided by the approval rate of the largest group on that attribute. Equal-opportunity difference is the gap in approval rate among accounts that did not default (group minus reference). Sex reference: `{fairness['reference']['sex']}`. Age-band reference: `{fairness['reference']['age_band']}`.

Minimum AIR: {metrics['metrics']['fairness']['air_min']:.4f}. Maximum AIR: {metrics['metrics']['fairness']['air_max']:.4f}.

{flag_lines}

These ratios describe this test sample. They are not a legal determination of disparate impact.

## Population stability

PSI compares the training distribution with the test distribution. Edges are the training quantiles. On a random split, PSI should be small. The largest value in this run is {highest_psi['feature']} at {highest_psi['psi']:.4f}.

| Feature | PSI | Bins |
|---|---:|---:|
{psi_lines}

A common reading is PSI below 0.10 as stable, 0.10 to 0.25 as a moderate shift, and above 0.25 as a large shift. This dataset has no application date, so the check is a split diagnostic rather than a through-the-time monitor.

## Recommendation

Use `{name}` as the champion and the weight-of-evidence scorecard as the challenger on this dataset, based on validation AUROC. Keep the scorecard as the interpretable benchmark: its points table is the audit trail for the linear model. Do not deploy either model for credit decisions. Any real use would need a time-based validation sample, a documented adverse-action policy, and a review of the fairness flags above.

## Limitations

- The data are Taiwanese credit cards from 2005, not a current US portfolio.
- There is no reject inference and no application or behavioral timeline.
- Monotone constraints are modeling choices. They can hide a real non-monotone pattern.
- Sex and age are not inputs, but other features can proxy for them. The fairness table is the check, not a fix.
- Isotonic calibration and early stopping both see the validation labels, so validation metrics are optimistic. Test metrics are the ones to quote.
- Reason codes explain this model. They are not a compliance sign-off.
"""


def apply_coverage(metrics: dict, percent: float) -> dict:
    """Record pytest-cov's total for src/ without inventing the number."""
    updated = dict(metrics)
    criteria = [item for item in metrics["success_criteria"] if item["name"] != "Test coverage of src/"]
    criteria.append(criterion("Test coverage of src/", percent, ">=", 75.0))
    updated["success_criteria"] = criteria
    return updated
