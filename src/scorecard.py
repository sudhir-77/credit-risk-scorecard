"""Weight-of-evidence binning and a logistic regression scorecard.

Points use a base score of 600 and 20 points to double the odds of non-default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data import SEED
from src.metrics import auroc

PDO = 20.0
BASE_SCORE = 600.0


def factor_pdo(pdo: float = PDO) -> float:
    return float(pdo / np.log(2.0))


def score_from_logit(logit: float, base: float = BASE_SCORE, pdo: float = PDO) -> float:
    """Map a default log-odds to points. Odds of non-default double when score rises by `pdo`."""
    return float(base - factor_pdo(pdo) * logit)


def woe_from_counts(
    n_good: float,
    n_bad: float,
    total_good: float,
    total_bad: float,
    n_bins: int,
) -> float:
    """Smoothed WoE = ln(share of non-defaults / share of defaults)."""
    goods = (n_good + 0.5) / (total_good + 0.5 * n_bins)
    bads = (n_bad + 0.5) / (total_bad + 0.5 * n_bins)
    return float(np.log(goods / bads))


@dataclass
class FeatureBins:
    feature: str
    kind: str
    labels: list[str]
    woe: list[float]
    n_good: list[int]
    n_bad: list[int]
    iv: float
    edges: list[float] = field(default_factory=list)
    members: list[list[float]] = field(default_factory=list)


def _initial_bins(
    values: np.ndarray,
    y: np.ndarray,
    max_bins: int,
) -> tuple[str, list[dict], list[float], list[float]]:
    unique = np.unique(values)
    if len(unique) <= 12:
        bins = []
        for level in unique:
            mask = values == level
            bins.append(
                {
                    "lo": float(level),
                    "hi": float(level),
                    "n_good": int(np.sum(mask & (y == 0))),
                    "n_bad": int(np.sum(mask & (y == 1))),
                    "members": [float(level)],
                }
            )
        return "discrete", bins, [], []

    quantiles = np.linspace(0.0, 1.0, max_bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 2:
        edges = np.array([float(values.min()) - 1.0, float(values.max()) + 1.0])
    inner = edges[1:-1]
    indexes = np.digitize(values, inner, right=False) if len(inner) else np.zeros(len(values), dtype=int)
    bins = []
    for index in range(len(edges) - 1):
        mask = indexes == index
        bins.append(
            {
                "lo": float(edges[index]),
                "hi": float(edges[index + 1]),
                "n_good": int(np.sum(mask & (y == 0))),
                "n_bad": int(np.sum(mask & (y == 1))),
                "members": [],
            }
        )
    return "interval", bins, [float(edge) for edge in edges], []


def _merge_pair(bins: list[dict], index: int) -> list[dict]:
    left, right = bins[index], bins[index + 1]
    merged = {
        "lo": left["lo"],
        "hi": right["hi"],
        "n_good": left["n_good"] + right["n_good"],
        "n_bad": left["n_bad"] + right["n_bad"],
        "members": [*left.get("members", []), *right.get("members", [])],
    }
    return [*bins[:index], merged, *bins[index + 2 :]]


def merge_rare_bins(bins: list[dict], min_count: int) -> list[dict]:
    guard = 0
    while len(bins) > 1 and guard < 100:
        guard += 1
        sizes = [item["n_good"] + item["n_bad"] for item in bins]
        smallest = int(np.argmin(sizes))
        if sizes[smallest] >= min_count:
            break
        if smallest == 0:
            neighbor = 0
        elif smallest == len(bins) - 1:
            neighbor = smallest - 1
        else:
            left_size = sizes[smallest - 1]
            right_size = sizes[smallest + 1]
            neighbor = smallest - 1 if left_size <= right_size else smallest
        bins = _merge_pair(bins, neighbor)
    return bins


def pool_monotone_bins(
    bins: list[dict],
    monotone: int,
    total_good: int,
    total_bad: int,
) -> list[dict]:
    """Pool adjacent bins until WoE follows the monotonic constraint.

    Constraint +1 (higher x, higher risk) requires non-increasing WoE.
    Constraint -1 requires non-decreasing WoE.
    """
    if monotone == 0 or len(bins) < 2:
        return bins
    want_nonincreasing = monotone > 0
    guard = 0
    while len(bins) > 1 and guard < 100:
        guard += 1
        woes = [
            woe_from_counts(item["n_good"], item["n_bad"], total_good, total_bad, len(bins))
            for item in bins
        ]
        violation = None
        for index in range(len(woes) - 1):
            if want_nonincreasing and woes[index] < woes[index + 1] - 1e-12:
                violation = index
                break
            if not want_nonincreasing and woes[index] > woes[index + 1] + 1e-12:
                violation = index
                break
        if violation is None:
            break
        bins = _merge_pair(bins, violation)
    return bins


def _finalize(feature: str, kind: str, bins: list[dict], edges: list[float], total_good: int, total_bad: int) -> FeatureBins:
    labels: list[str] = []
    woes: list[float] = []
    n_good: list[int] = []
    n_bad: list[int] = []
    iv = 0.0
    n_bins = max(len(bins), 1)
    for item in bins:
        if kind == "discrete":
            labels.append(",".join(f"{member:g}" for member in item["members"]))
        else:
            labels.append(f"({item['lo']:.6g}, {item['hi']:.6g}]")
        woe = woe_from_counts(item["n_good"], item["n_bad"], total_good, total_bad, n_bins)
        goods = (item["n_good"] + 0.5) / (total_good + 0.5 * n_bins)
        bads = (item["n_bad"] + 0.5) / (total_bad + 0.5 * n_bins)
        iv += (goods - bads) * woe
        woes.append(woe)
        n_good.append(int(item["n_good"]))
        n_bad.append(int(item["n_bad"]))
    members = [list(item.get("members", [])) for item in bins] if kind == "discrete" else []
    return FeatureBins(
        feature=feature,
        kind=kind,
        labels=labels,
        woe=woes,
        n_good=n_good,
        n_bad=n_bad,
        iv=float(iv),
        edges=edges if kind == "interval" else [],
        members=members,
    )


def fit_feature_bins(
    values: np.ndarray,
    y: np.ndarray,
    feature: str,
    monotone: int,
    max_bins: int,
    min_count: int,
) -> FeatureBins:
    values = np.asarray(values, dtype=float)
    y = np.asarray(y, dtype=int)
    total_good = int(np.sum(y == 0))
    total_bad = int(np.sum(y == 1))
    if total_good == 0 or total_bad == 0:
        raise ValueError(f"Cannot bin {feature} with a single class")
    kind, bins, edges, _levels = _initial_bins(values, y, max_bins=max_bins)
    bins = merge_rare_bins(bins, min_count=min_count)
    bins = pool_monotone_bins(bins, monotone=monotone, total_good=total_good, total_bad=total_bad)
    # Interval edges are rebuilt from the pooled bins so transform stays aligned.
    if kind == "interval":
        edges = [bins[0]["lo"], *[item["hi"] for item in bins]]
    return _finalize(feature, kind, bins, edges, total_good, total_bad)


def transform_feature(values: np.ndarray, spec: FeatureBins) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    woe = np.zeros(len(values), dtype=float)
    if spec.kind == "discrete":
        lookup: dict[float, float] = {}
        for members, weight in zip(spec.members, spec.woe):
            for level in members:
                lookup[float(level)] = weight
        for index, value in enumerate(values):
            woe[index] = lookup.get(float(value), 0.0)
        return woe
    edges = np.asarray(spec.edges, dtype=float)
    inner = edges[1:-1]
    if len(inner) == 0:
        woe[:] = spec.woe[0]
        return woe
    indexes = np.digitize(values, inner, right=False)
    indexes = np.clip(indexes, 0, len(spec.woe) - 1)
    return np.asarray(spec.woe, dtype=float)[indexes]


class ScorecardModel:
    """WoE encoder plus L2 logistic regression. Binning is fit on train only."""

    def __init__(self, monotone: dict[str, int], max_bins: int = 8, min_count: int = 50, seed: int = SEED):
        self.monotone = monotone
        self.max_bins = max_bins
        self.min_count = min_count
        self.seed = seed
        self.features_: list[str] = []
        self.bins_: dict[str, FeatureBins] = {}
        self.clf_: LogisticRegression | None = None
        self.C_: float | None = None
        self.class_weight_: str | None = None

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        x_val: pd.DataFrame,
        y_val: pd.Series | np.ndarray,
    ) -> "ScorecardModel":
        self.features_ = list(x_train.columns)
        y_tr = np.asarray(y_train, dtype=int)
        y_va = np.asarray(y_val, dtype=int)
        self.bins_ = {}
        for feature in self.features_:
            self.bins_[feature] = fit_feature_bins(
                x_train[feature].to_numpy(),
                y_tr,
                feature=feature,
                monotone=int(self.monotone.get(feature, 0)),
                max_bins=self.max_bins,
                min_count=self.min_count,
            )
        z_train = self.transform(x_train)
        z_val = self.transform(x_val)
        best_auc = -np.inf
        best: LogisticRegression | None = None
        for C in (0.3, 1.0, 3.0):
            for class_weight in (None, "balanced"):
                clf = LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    solver="lbfgs",
                    max_iter=500,
                    random_state=self.seed,
                )
                clf.fit(z_train, y_tr)
                probabilities = clf.predict_proba(z_val)[:, 1]
                auc = auroc(y_va, probabilities)
                if auc > best_auc:
                    best_auc = auc
                    best = clf
                    self.C_ = C
                    self.class_weight_ = class_weight
        if best is None:
            raise RuntimeError("Scorecard search did not fit a model")
        self.clf_ = best
        self.val_auroc_ = float(best_auc)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns = [transform_feature(frame[feature].to_numpy(), self.bins_[feature]) for feature in self.features_]
        return np.column_stack(columns)

    def predict_logit(self, frame: pd.DataFrame) -> np.ndarray:
        if self.clf_ is None:
            raise RuntimeError("Scorecard is not fit")
        matrix = self.transform(frame)
        return self.clf_.decision_function(matrix)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.clf_ is None:
            raise RuntimeError("Scorecard is not fit")
        return self.clf_.predict_proba(self.transform(frame))[:, 1]

    def point_scores(self, frame: pd.DataFrame) -> np.ndarray:
        logits = self.predict_logit(frame)
        return BASE_SCORE - factor_pdo(PDO) * logits

    def points_table(self) -> pd.DataFrame:
        if self.clf_ is None:
            raise RuntimeError("Scorecard is not fit")
        factor = factor_pdo(PDO)
        intercept = float(self.clf_.intercept_[0])
        rows = [
            {
                "feature": "__INTERCEPT__",
                "bin": "base",
                "woe": np.nan,
                "coefficient": intercept,
                "points": score_from_logit(0.0) - factor * intercept,
                "n_good": np.nan,
                "n_bad": np.nan,
                "iv": np.nan,
            }
        ]
        # score = (base - factor * intercept) + sum(-factor * coef * woe)
        # and score_from_logit(intercept + sum(coef*woe)) matches that sum.
        coefficients = self.clf_.coef_.ravel()
        for index, feature in enumerate(self.features_):
            spec = self.bins_[feature]
            coef = float(coefficients[index])
            for label, woe, n_good, n_bad in zip(spec.labels, spec.woe, spec.n_good, spec.n_bad):
                rows.append(
                    {
                        "feature": feature,
                        "bin": label,
                        "woe": woe,
                        "coefficient": coef,
                        "points": float(-factor * coef * woe),
                        "n_good": n_good,
                        "n_bad": n_bad,
                        "iv": spec.iv,
                    }
                )
        return pd.DataFrame(rows)
