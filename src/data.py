"""Download, validate, and split the UCI credit-card default data."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SEED = 42
EXPECTED_ROWS = 30_000
EXPECTED_DEFAULTS = 6_636
EXPECTED_MD5 = "25a64500f543274fd6c9db905bfeb968"
DATA_FILENAME = "default_of_credit_card_clients.xls"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)

PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]
ENGINEERED = ["UTIL_1", "AVG_PAY_STATUS", "MAX_PAY_STATUS", "AVG_PAY_AMT"]
RAW_FEATURES = ["LIMIT_BAL", "EDUCATION", "MARRIAGE", *PAY_STATUS, *BILL_AMT, *PAY_AMT]
MODEL_FEATURES = [*RAW_FEATURES, *ENGINEERED]

# +1 means a higher value cannot decrease predicted default risk.
MONOTONE_CONSTRAINTS: dict[str, int] = {
    "LIMIT_BAL": -1,
    "EDUCATION": 0,
    "MARRIAGE": 0,
    **{column: 1 for column in PAY_STATUS},
    **{column: 0 for column in BILL_AMT},
    **{column: -1 for column in PAY_AMT},
    "UTIL_1": 1,
    "AVG_PAY_STATUS": 1,
    "MAX_PAY_STATUS": 1,
    "AVG_PAY_AMT": -1,
}

AGE_BANDS = ["<30", "30-39", "40+"]
SEX_LABELS = ["female", "male"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def constraint_vector(columns: list[str]) -> list[int]:
    missing = [column for column in columns if column not in MONOTONE_CONSTRAINTS]
    if missing:
        raise KeyError(f"No monotonic constraint registered for {missing}")
    return [int(MONOTONE_CONSTRAINTS[column]) for column in columns]


def age_band(age: pd.Series) -> pd.Series:
    bands = pd.Series(pd.NA, index=age.index, dtype="object")
    bands.loc[age < 30] = "<30"
    bands.loc[(age >= 30) & (age <= 39)] = "30-39"
    bands.loc[age >= 40] = "40+"
    if bands.isna().any():
        raise ValueError("Age values fell outside the three bands")
    return bands


def sex_label(sex: pd.Series) -> pd.Series:
    labels = sex.map({1: "male", 2: "female"})
    if labels.isna().any():
        unknown = sorted(set(sex.loc[labels.isna()].tolist()))
        raise ValueError(f"Unexpected SEX codes: {unknown}")
    return labels


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(data_dir: Path) -> Path:
    """Return the cached xls, downloading it when the checksum does not match."""
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / DATA_FILENAME
    if dest.exists() and file_md5(dest) == EXPECTED_MD5:
        return dest
    if dest.exists():
        dest.unlink()
    try:
        with requests.get(DATA_URL, timeout=120, stream=True) as response:
            response.raise_for_status()
            with dest.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    if chunk:
                        handle.write(chunk)
    except requests.RequestException as exc:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Could not download the UCI dataset: {exc}") from exc
    actual = file_md5(dest)
    if actual != EXPECTED_MD5:
        dest.unlink()
        raise ValueError(f"Checksum mismatch: expected {EXPECTED_MD5}, got {actual}")
    return dest


def load_raw(path: Path) -> pd.DataFrame:
    """Load the UCI workbook. Row 0 is the X1..Y header; row 1 holds names."""
    import xlrd

    book = xlrd.open_workbook(str(path))
    sheet = book.sheet_by_index(0)
    headers = [str(sheet.cell_value(1, col)).strip() for col in range(sheet.ncols)]
    rows = [
        [sheet.cell_value(row, col) for col in range(sheet.ncols)]
        for row in range(2, sheet.nrows)
    ]
    frame = pd.DataFrame(rows, columns=headers)
    frame = frame.rename(columns={"default payment next month": "label"})
    expected = [
        "ID",
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        *PAY_STATUS,
        *BILL_AMT,
        *PAY_AMT,
        "label",
    ]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    for column in expected:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["ID"] = frame["ID"].astype(int)
    frame["label"] = frame["label"].astype(int)
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(frame)}")
    defaults = int(frame["label"].sum())
    if defaults != EXPECTED_DEFAULTS:
        raise ValueError(f"Expected {EXPECTED_DEFAULTS} defaults, found {defaults}")
    if frame["ID"].duplicated().any():
        raise ValueError("Duplicate client IDs")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("Label column is not binary")
    return frame.loc[:, expected].copy()


def add_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    limit = out["LIMIT_BAL"].clip(lower=1.0)
    out["UTIL_1"] = (out["BILL_AMT1"] / limit).clip(-0.5, 2.0)
    out["AVG_PAY_STATUS"] = out[PAY_STATUS].mean(axis=1)
    out["MAX_PAY_STATUS"] = out[PAY_STATUS].max(axis=1)
    out["AVG_PAY_AMT"] = out[PAY_AMT].mean(axis=1)
    return out


def attach_groups(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["sex"] = sex_label(out["SEX"])
    out["age_band"] = age_band(out["AGE"])
    return out


def subsample(frame: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("label", sort=False):
        count = min(len(group), max(1, int(round(len(group) * frac))))
        parts.append(group.sample(n=count, random_state=seed))
    out = pd.concat(parts, axis=0)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_splits(
    frame: pd.DataFrame,
    seed: int = SEED,
    test_size: float = 0.2,
    val_fraction_of_rest: float = 0.25,
) -> dict[str, pd.DataFrame]:
    """Stratified 60/20/20 split. 0.25 of the post-test remainder is 20% overall."""
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame["label"],
    )
    train, validation = train_test_split(
        train_val,
        test_size=val_fraction_of_rest,
        random_state=seed,
        stratify=train_val["label"],
    )
    splits = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    id_sets = {name: set(part["ID"].tolist()) for name, part in splits.items()}
    overlap = (
        (id_sets["train"] & id_sets["validation"])
        | (id_sets["train"] & id_sets["test"])
        | (id_sets["validation"] & id_sets["test"])
    )
    if overlap:
        raise RuntimeError("Train, validation, and test IDs overlap")
    return splits


def load_modeling_frame(data_dir: Path, quick: bool, seed: int = SEED) -> pd.DataFrame:
    path = download_dataset(data_dir)
    frame = attach_groups(add_model_features(load_raw(path)))
    protected = {"SEX", "AGE", "sex", "age_band", "label", "ID"}
    leaked = protected.intersection(MODEL_FEATURES)
    if leaked:
        raise RuntimeError(f"Protected columns entered the model matrix: {sorted(leaked)}")
    if quick:
        frame = subsample(frame, frac=0.12, seed=seed)
    return frame


def split_id_payload(splits: dict[str, pd.DataFrame]) -> dict[str, list[int]]:
    return {
        "train": [int(value) for value in splits["train"]["ID"].tolist()],
        "validation": [int(value) for value in splits["validation"]["ID"].tolist()],
        "test": [int(value) for value in splits["test"]["ID"].tolist()],
    }


def assert_split_integrity(payload: dict[str, list[int]]) -> None:
    train, validation, test = (
        set(payload["train"]),
        set(payload["validation"]),
        set(payload["test"]),
    )
    if train & validation or train & test or validation & test:
        raise AssertionError("Split IDs overlap")
    if len(train) != len(payload["train"]):
        raise AssertionError("Duplicate train IDs")
    if len(validation) != len(payload["validation"]):
        raise AssertionError("Duplicate validation IDs")
    if len(test) != len(payload["test"]):
        raise AssertionError("Duplicate test IDs")
    if not train or not validation or not test:
        raise AssertionError("A split is empty")


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in MODEL_FEATURES if column not in frame.columns]
    if missing:
        raise KeyError(f"Feature matrix is missing {missing}")
    matrix = frame.loc[:, MODEL_FEATURES].astype(np.float64)
    if matrix.isna().any().any():
        raise ValueError("Model matrix contains missing values")
    return matrix
