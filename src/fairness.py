"""Adverse impact ratio and equal-opportunity difference.

Approval means the calibrated probability of default is below the cutoff.
Equal opportunity uses approvals among accounts that did not default (label 0).
The reference group is the largest group on that attribute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AIR_LOW = 0.80
AIR_HIGH = 1.25


def _group_stats(
    y: np.ndarray,
    approve: np.ndarray,
    groups: pd.Series,
    level: str,
) -> dict[str, float | int]:
    mask = (groups == level).to_numpy()
    n = int(mask.sum())
    if n == 0:
        return {
            "n": 0,
            "n_good": 0,
            "approval_rate": float("nan"),
            "tpr_good": float("nan"),
        }
    y_g = y[mask]
    approve_g = approve[mask]
    goods = y_g == 0
    n_good = int(goods.sum())
    tpr = float(approve_g[goods].mean()) if n_good else float("nan")
    return {
        "n": n,
        "n_good": n_good,
        "approval_rate": float(approve_g.mean()),
        "tpr_good": tpr,
    }


def fairness_report(
    y_true: np.ndarray,
    scores: np.ndarray,
    sex: pd.Series,
    age_band: pd.Series,
    cutoff: float,
    group_order: dict[str, list[str]] | None = None,
) -> dict:
    y = np.asarray(y_true, dtype=int)
    approve = np.asarray(scores, dtype=float) < float(cutoff)
    order = group_order or {
        "sex": ["female", "male"],
        "age_band": ["<30", "30-39", "40+"],
    }
    series_map = {"sex": sex.reset_index(drop=True), "age_band": age_band.reset_index(drop=True)}
    report: dict = {"cutoff": float(cutoff), "flags": [], "reference": {}}
    for attribute, levels in order.items():
        groups = series_map[attribute]
        counts = {level: int((groups == level).sum()) for level in levels}
        reference = max(levels, key=lambda level: (counts[level], level))
        report["reference"][attribute] = reference
        ref_stats = _group_stats(y, approve, groups, reference)
        report[attribute] = {}
        for level in levels:
            stats = _group_stats(y, approve, groups, level)
            if ref_stats["n"] == 0 or not np.isfinite(ref_stats["approval_rate"]) or ref_stats["approval_rate"] == 0:
                air = float("nan")
            else:
                air = float(stats["approval_rate"]) / float(ref_stats["approval_rate"])
            if np.isfinite(stats["tpr_good"]) and np.isfinite(ref_stats["tpr_good"]):
                eod = float(stats["tpr_good"]) - float(ref_stats["tpr_good"])
            else:
                eod = float("nan")
            payload = {
                "n": int(stats["n"]),
                "n_good": int(stats["n_good"]),
                "approval_rate": float(stats["approval_rate"]),
                "tpr_good": float(stats["tpr_good"]),
                "air": float(air),
                "equal_opportunity_difference": float(eod),
                "is_reference": level == reference,
            }
            report[attribute][level] = payload
            if np.isfinite(air) and (air < AIR_LOW or air > AIR_HIGH):
                report["flags"].append(
                    {
                        "attribute": attribute,
                        "group": level,
                        "air": float(air),
                        "message": (
                            f"FLAG: AIR for {attribute}={level} is {air:.3f}, outside 0.80-1.25."
                        ),
                    }
                )
    return report


def fairness_is_complete(report: dict) -> bool:
    if "cutoff" not in report:
        return False
    sex = report.get("sex", {})
    age = report.get("age_band", {})
    if set(sex) != {"female", "male"}:
        return False
    if set(age) != {"<30", "30-39", "40+"}:
        return False
    for block in (sex, age):
        for stats in block.values():
            if "air" not in stats or "equal_opportunity_difference" not in stats:
                return False
            if not np.isfinite(stats["air"]) or not np.isfinite(stats["equal_opportunity_difference"]):
                return False
    return True
