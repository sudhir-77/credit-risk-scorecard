"""Loader checks against the cached UCI workbook."""

from pathlib import Path

from src.data import EXPECTED_DEFAULTS, EXPECTED_MD5, EXPECTED_ROWS, download_dataset, file_md5, load_raw


def test_cached_workbook_checksum_rows_and_defaults():
    path = download_dataset(Path("data"))
    assert file_md5(path) == EXPECTED_MD5
    frame = load_raw(path)
    assert len(frame) == EXPECTED_ROWS
    assert int(frame["label"].sum()) == EXPECTED_DEFAULTS
    assert frame["ID"].is_unique
