"""Monotonic LightGBM, isotonic calibration, and TreeSHAP contributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.isotonic import IsotonicRegression

from src.data import SEED, constraint_vector
from src.metrics import auroc


def candidate_params(quick: bool) -> list[dict]:
    full = [
        {
            "max_depth": 3,
            "num_leaves": 8,
            "min_child_samples": 100,
            "learning_rate": 0.08,
            "n_estimators": 300,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        {
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 80,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
        {
            "max_depth": 5,
            "num_leaves": 24,
            "min_child_samples": 60,
            "learning_rate": 0.05,
            "n_estimators": 450,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
        },
    ]
    quick_params = [
        {
            "max_depth": 3,
            "num_leaves": 8,
            "min_child_samples": 20,
            "learning_rate": 0.1,
            "n_estimators": 40,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        }
    ]
    if quick:
        return quick_params
    return full


class MonotonicGBM:
    """LightGBM with monotone constraints. Hyperparameters are chosen on validation AUROC."""

    def __init__(self, quick: bool = False, seed: int = SEED):
        self.quick = quick
        self.seed = seed
        self.model: LGBMClassifier | None = None
        self.features_: list[str] = []
        self.constraints_: list[int] = []
        self.best_params_: dict | None = None
        self.best_iteration_: int | None = None
        self.scale_pos_weight_: float | None = None
        self.val_auroc_: float | None = None

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        x_val: pd.DataFrame,
        y_val: pd.Series | np.ndarray,
    ) -> "MonotonicGBM":
        self.features_ = list(x_train.columns)
        self.constraints_ = constraint_vector(self.features_)
        y_tr = np.asarray(y_train, dtype=int)
        y_va = np.asarray(y_val, dtype=int)
        positives = max(int(y_tr.sum()), 1)
        negatives = max(int(len(y_tr) - positives), 1)
        balanced = negatives / positives
        weights = (balanced,) if self.quick else (1.0, balanced)
        best_auc = -np.inf
        best_model: LGBMClassifier | None = None
        for params in candidate_params(self.quick):
            for weight in weights:
                model = _fit_one(
                    x_train,
                    y_tr,
                    x_val,
                    y_va,
                    params,
                    self.constraints_,
                    scale_pos_weight=float(weight),
                    seed=self.seed,
                )
                probabilities = predict_proba(model, x_val)
                auc = auroc(y_va, probabilities)
                if auc > best_auc:
                    best_auc = auc
                    best_model = model
                    self.best_params_ = {**params, "scale_pos_weight": float(weight)}
                    iteration = int(getattr(model, "best_iteration_", 0) or params["n_estimators"])
                    self.best_iteration_ = iteration
                    self.scale_pos_weight_ = float(weight)
        if best_model is None or self.best_iteration_ is None:
            raise RuntimeError("LightGBM search did not fit a model")
        self.model = best_model
        self.val_auroc_ = float(best_auc)
        return self

    def predict_margin(self, frame: pd.DataFrame) -> np.ndarray:
        booster, iteration = self._booster()
        return np.asarray(
            booster.predict(frame[self.features_], raw_score=True, num_iteration=iteration),
            dtype=float,
        )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LightGBM is not fit")
        return predict_proba(self.model, frame[self.features_])

    def shap_values(self, frame: pd.DataFrame) -> np.ndarray:
        """TreeSHAP contributions on the default log-odds, excluding the expected value."""
        booster, iteration = self._booster()
        contrib = booster.predict(
            frame[self.features_],
            pred_contrib=True,
            num_iteration=iteration,
        )
        contrib = np.asarray(contrib, dtype=float)
        if contrib.ndim != 2 or contrib.shape[1] != len(self.features_) + 1:
            raise RuntimeError(f"Unexpected contribution shape {contrib.shape}")
        return contrib[:, :-1]

    def _booster(self):
        if self.model is None or self.best_iteration_ is None:
            raise RuntimeError("LightGBM is not fit")
        return self.model.booster_, self.best_iteration_


def _fit_one(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict,
    constraints: list[int],
    scale_pos_weight: float,
    seed: int,
) -> LGBMClassifier:
    model = LGBMClassifier(
        objective="binary",
        max_depth=params["max_depth"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        subsample=params["subsample"],
        subsample_freq=1 if params["subsample"] < 1 else 0,
        colsample_bytree=params["colsample_bytree"],
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=constraints,
        random_state=seed,
        n_jobs=4,
        verbose=-1,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="auc",
        callbacks=[
            early_stopping(stopping_rounds=30, verbose=False),
            log_evaluation(period=0),
        ],
    )
    return model


def predict_proba(model: LGBMClassifier, frame: pd.DataFrame) -> np.ndarray:
    iteration = getattr(model, "best_iteration_", None)
    if iteration:
        return model.predict_proba(frame, num_iteration=iteration)[:, 1]
    return model.predict_proba(frame)[:, 1]


class IsotonicCalibrator:
    """Isotonic regression fit on validation scores only."""

    def __init__(self) -> None:
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.model.fit(np.asarray(scores, dtype=float), np.asarray(y, dtype=float))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(scores, dtype=float)), dtype=float)
