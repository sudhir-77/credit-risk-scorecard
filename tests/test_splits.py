"""Split integrity and protected-attribute exclusion."""

import numpy as np
import pandas as pd
import pytest

from src.data import (
    AGE_BANDS,
    MODEL_FEATURES,
    MONOTONE_CONSTRAINTS,
    age_band,
    assert_split_integrity,
    feature_matrix,
    make_splits,
    sex_label,
    subsample,
)


def _frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, n)
    sex = rng.integers(1, 3, n)
    label = rng.integers(0, 2, n)
    # Force both classes.
    label[:10] = 0
    label[10:20] = 1
    data = {
        "ID": np.arange(1, n + 1),
        "SEX": sex,
        "AGE": age,
        "label": label,
    }
    for column in MODEL_FEATURES:
        data[column] = rng.normal(size=n)
    frame = pd.DataFrame(data)
    frame["sex"] = sex_label(frame["SEX"])
    frame["age_band"] = age_band(frame["AGE"])
    return frame


def test_splits_are_disjoint_and_cover_every_id():
    frame = _frame()
    splits = make_splits(frame, seed=7)
    payload = {name: part["ID"].tolist() for name, part in splits.items()}
    assert_split_integrity(payload)
    covered = set(payload["train"]) | set(payload["validation"]) | set(payload["test"])
    assert covered == set(frame["ID"])
    total = sum(len(part) for part in splits.values())
    assert total == len(frame)
    # 60/20/20 within rounding of the stratified splitter.
    assert len(splits["test"]) == pytest.approx(0.2 * len(frame), abs=2)
    assert len(splits["validation"]) == pytest.approx(0.2 * len(frame), abs=2)


def test_feature_matrix_drops_sex_and_age():
    frame = _frame(80)
    matrix = feature_matrix(frame)
    assert list(matrix.columns) == MODEL_FEATURES
    for banned in ("SEX", "AGE", "sex", "age_band", "label", "ID"):
        assert banned not in matrix.columns
    assert set(MONOTONE_CONSTRAINTS) == set(MODEL_FEATURES)


def test_age_bands_and_sex_labels():
    bands = age_band(pd.Series([21, 29, 30, 39, 40, 75]))
    assert bands.tolist() == ["<30", "<30", "30-39", "30-39", "40+", "40+"]
    assert set(AGE_BANDS) == {"<30", "30-39", "40+"}
    assert sex_label(pd.Series([1, 2, 1])).tolist() == ["male", "female", "male"]
    with pytest.raises(ValueError):
        sex_label(pd.Series([1, 3]))


def test_subsample_keeps_both_classes():
    frame = _frame(500, seed=1)
    sample = subsample(frame, frac=0.12, seed=1)
    assert sample["label"].nunique() == 2
    assert len(sample) < len(frame)
    assert set(sample["ID"]).issubset(set(frame["ID"]))
