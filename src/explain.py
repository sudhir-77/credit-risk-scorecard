"""Adverse-action reason codes from TreeSHAP and monotonic constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd

REFERENCE_LEVELS = (
    "median",
    "favorable_quartile",
    "favorable_decile",
    "favorable_extreme",
)


def training_references(frame: pd.DataFrame, constraints: dict[str, int]) -> dict[str, dict[str, float]]:
    """Reference levels used only to decide whether a value is on the adverse side.

    The favorable side is the low tail for a +1 constraint and the high tail for a -1 constraint.
    """
    references: dict[str, dict[str, float]] = {}
    for column in frame.columns:
        series = frame[column].astype(float)
        constraint = int(constraints.get(column, 0))
        median = float(series.median())
        if constraint > 0:
            references[column] = {
                "median": median,
                "favorable_quartile": float(series.quantile(0.25)),
                "favorable_decile": float(series.quantile(0.10)),
                "favorable_extreme": float(series.min()),
            }
        elif constraint < 0:
            references[column] = {
                "median": median,
                "favorable_quartile": float(series.quantile(0.75)),
                "favorable_decile": float(series.quantile(0.90)),
                "favorable_extreme": float(series.max()),
            }
        else:
            references[column] = {level: median for level in REFERENCE_LEVELS}
    return references


def adverse_side(value: float, reference: float, constraint: int) -> bool:
    if constraint > 0:
        return value > reference
    if constraint < 0:
        return value < reference
    return False


def reason_code(feature: str, constraint: int) -> str:
    if constraint > 0:
        return f"{feature}_HIGH"
    if constraint < 0:
        return f"{feature}_LOW"
    raise ValueError(f"{feature} has no monotonic direction")


def sign_agrees(code: str, constraint: int, value: float, reference: float) -> bool:
    """The HIGH/LOW token matches the constraint and the value is on that side of the reference."""
    if constraint > 0:
        return code.endswith("_HIGH") and value > reference
    if constraint < 0:
        return code.endswith("_LOW") and value < reference
    return False


def _candidates(
    values: np.ndarray,
    shap_row: np.ndarray,
    columns: list[str],
    constraints: dict[str, int],
    references: dict[str, dict[str, float]],
    level: str,
    used: set[str],
    require_positive_shap: bool,
) -> list[tuple[float, str, str, float]]:
    found: list[tuple[float, str, str, float]] = []
    for index, column in enumerate(columns):
        if column in used:
            continue
        constraint = int(constraints.get(column, 0))
        if constraint == 0:
            continue
        reference = float(references[column][level])
        value = float(values[index])
        contribution = float(shap_row[index])
        if not adverse_side(value, reference, constraint):
            continue
        if require_positive_shap and contribution <= 0:
            continue
        found.append((contribution, column, level, reference))
    found.sort(key=lambda item: item[0], reverse=True)
    return found


def select_reason_codes(
    values: np.ndarray,
    shap_row: np.ndarray,
    columns: list[str],
    constraints: dict[str, int],
    references: dict[str, dict[str, float]],
    k: int = 4,
) -> list[dict]:
    """Top-k adverse reasons ranked by TreeSHAP.

    A code is kept only when the applicant's value sits on the adverse side of a
    training reference, so the HIGH/LOW direction matches the monotonic constraint.
    Positive contributions are preferred. The reference relaxes from the median
    toward a favorable benchmark only when fewer than k positive contributions exist.
    """
    chosen: list[dict] = []
    used: set[str] = set()
    for level in REFERENCE_LEVELS:
        for contribution, column, used_level, reference in _candidates(
            values,
            shap_row,
            columns,
            constraints,
            references,
            level,
            used,
            require_positive_shap=True,
        ):
            if len(chosen) >= k:
                break
            constraint = int(constraints[column])
            chosen.append(
                {
                    "feature": column,
                    "code": reason_code(column, constraint),
                    "shap": contribution,
                    "constraint": constraint,
                    "value": float(values[columns.index(column)]),
                    "reference": reference,
                    "reference_level": used_level,
                }
            )
            used.add(column)
        if len(chosen) >= k:
            break
    if len(chosen) < k:
        for contribution, column, used_level, reference in _candidates(
            values,
            shap_row,
            columns,
            constraints,
            references,
            "favorable_extreme",
            used,
            require_positive_shap=False,
        ):
            if len(chosen) >= k:
                break
            constraint = int(constraints[column])
            chosen.append(
                {
                    "feature": column,
                    "code": reason_code(column, constraint),
                    "shap": contribution,
                    "constraint": constraint,
                    "value": float(values[columns.index(column)]),
                    "reference": reference,
                    "reference_level": used_level,
                }
            )
            used.add(column)
    if len(chosen) < k:
        raise RuntimeError(f"Only found {len(chosen)} reason codes; expected {k}")
    for item in chosen:
        item["sign_agrees"] = bool(
            sign_agrees(item["code"], item["constraint"], item["value"], item["reference"])
        )
    return chosen[:k]


def build_reason_tables(
    ids: np.ndarray,
    frame: pd.DataFrame,
    shap_values: np.ndarray,
    declined: np.ndarray,
    constraints: dict[str, int],
    references: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(frame.columns)
    matrix = frame.to_numpy(dtype=float)
    code_rows: list[dict] = []
    audit_rows: list[dict] = []
    declined_ids = ids[declined]
    declined_x = matrix[declined]
    declined_shap = shap_values[declined]
    for client_id, values, shap_row in zip(declined_ids, declined_x, declined_shap):
        reasons = select_reason_codes(values, shap_row, columns, constraints, references, k=4)
        code_rows.append(
            {
                "id": int(client_id),
                "reason_1": reasons[0]["code"],
                "reason_2": reasons[1]["code"],
                "reason_3": reasons[2]["code"],
                "reason_4": reasons[3]["code"],
            }
        )
        for rank, reason in enumerate(reasons, start=1):
            audit_rows.append(
                {
                    "id": int(client_id),
                    "rank": rank,
                    "reason": reason["code"],
                    "feature": reason["feature"],
                    "shap": reason["shap"],
                    "constraint": reason["constraint"],
                    "value": reason["value"],
                    "reference": reason["reference"],
                    "reference_level": reason["reference_level"],
                    "sign_agrees": bool(reason["sign_agrees"]),
                }
            )
    codes = pd.DataFrame(code_rows, columns=["id", "reason_1", "reason_2", "reason_3", "reason_4"])
    audit = pd.DataFrame(audit_rows)
    if not codes.empty:
        codes = codes.sort_values("id").reset_index(drop=True)
        audit = audit.sort_values(["id", "rank"]).reset_index(drop=True)
    return codes, audit
