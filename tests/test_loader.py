"""
tests/test_loader.py
Pytest test suite for src/data/loader.py — Day 1.

Run with:
    pytest tests/test_loader.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytest

from src.data.loader import (
    CHURN_VALID_VALUES,
    EXPECTED_COLUMNS,
    ROW_COUNT_RANGE,
    load_raw_data,
)

# ── Fixture: path to the real raw dataset ─────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


# ── Helper: build a minimal valid CSV in-memory ───────────────────────────────

def _make_csv(
    n_rows: int = 7_043,
    columns: Optional[List[str]] = None,
    churn_values: Optional[List[str]] = None,
    total_charges_values: Optional[List[str]] = None,
) -> Path:
    """
    Write a synthetic CSV to a temp path and return it.
    Used to test validation logic without needing the real dataset.
    """
    cols = columns if columns is not None else EXPECTED_COLUMNS
    churn_col = churn_values if churn_values is not None else (["Yes", "No"] * (n_rows // 2 + 1))[:n_rows]
    tc_col = total_charges_values if total_charges_values is not None else ["1234.56"] * n_rows

    rows = []
    for i in range(n_rows):
        row = {c: "dummy" for c in cols}
        row["SeniorCitizen"]   = 0
        row["tenure"]          = 12
        row["MonthlyCharges"]  = 29.99
        row["TotalCharges"]    = tc_col[i]
        row["Churn"]           = churn_col[i]
        if "customerID" in row:
            row["customerID"]  = f"CUST-{i:05d}"
        rows.append(row)

    df = pd.DataFrame(rows, columns=cols)

    # Write to a temp file inside data/raw (already gitignored)
    tmp_path = PROJECT_ROOT / "data" / "raw" / "_test_synthetic.csv"
    df.to_csv(tmp_path, index=False)
    return tmp_path


@pytest.fixture(autouse=True)
def cleanup_synthetic(tmp_path_factory):
    """Remove any synthetic CSVs written during tests."""
    yield
    synthetic = PROJECT_ROOT / "data" / "raw" / "_test_synthetic.csv"
    if synthetic.exists():
        synthetic.unlink()


# ── Test 1: Real dataset loads without error ───────────────────────────────────

def test_real_dataset_loads():
    """
    The real downloaded CSV must load without raising any exception.
    Asserts: load_raw_data() returns a non-empty DataFrame.
    """
    if not RAW_CSV.exists():
        pytest.skip("Raw dataset not present — run scripts/download_data.py first.")

    df = load_raw_data(RAW_CSV)
    assert isinstance(df, pd.DataFrame), "load_raw_data() must return a DataFrame"
    assert not df.empty, "Loaded DataFrame must not be empty"


# ── Test 2: Shape is within expected bounds ────────────────────────────────────

def test_real_dataset_shape():
    """
    The dataset must have exactly 21 columns and a row count within [7000, 7500].
    Asserts: df.shape matches contract.
    """
    if not RAW_CSV.exists():
        pytest.skip("Raw dataset not present — run scripts/download_data.py first.")

    df = load_raw_data(RAW_CSV)
    n_rows, n_cols = df.shape

    assert n_cols == len(EXPECTED_COLUMNS), (
        f"Expected {len(EXPECTED_COLUMNS)} columns, got {n_cols}. "
        f"Columns present: {df.columns.tolist()}"
    )

    lo, hi = ROW_COUNT_RANGE
    assert lo <= n_rows <= hi, (
        f"Row count {n_rows} is outside expected range [{lo}, {hi}]."
    )


# ── Test 3: All expected columns are present ───────────────────────────────────

def test_expected_columns_present():
    """
    Every column in EXPECTED_COLUMNS must appear in the loaded DataFrame.
    Asserts: set(df.columns) == set(EXPECTED_COLUMNS).
    """
    if not RAW_CSV.exists():
        pytest.skip("Raw dataset not present — run scripts/download_data.py first.")

    df = load_raw_data(RAW_CSV)
    assert set(df.columns) == set(EXPECTED_COLUMNS), (
        f"Column mismatch.\n"
        f"Missing: {set(EXPECTED_COLUMNS) - set(df.columns)}\n"
        f"Extra:   {set(df.columns) - set(EXPECTED_COLUMNS)}"
    )


# ── Test 4: Churn column contains only 'Yes' and 'No' ─────────────────────────

def test_churn_column_values():
    """
    The 'Churn' column must contain only the values in CHURN_VALID_VALUES {'Yes', 'No'}.
    Asserts: no unexpected values exist in the column.
    """
    if not RAW_CSV.exists():
        pytest.skip("Raw dataset not present — run scripts/download_data.py first.")

    df = load_raw_data(RAW_CSV)
    actual_values = set(df["Churn"].dropna().unique())
    unexpected = actual_values - CHURN_VALID_VALUES
    assert not unexpected, (
        f"'Churn' column contains unexpected values: {unexpected}. "
        f"Valid values are: {CHURN_VALID_VALUES}"
    )


# ── Test 5 (Edge case): Missing columns raise descriptive ValueError ───────────

def test_missing_column_raises_value_error():
    """
    Edge case: a CSV missing the 'Churn' column must raise ValueError with a
    message that names the missing column — not a cryptic KeyError downstream.
    """
    cols_without_churn = [c for c in EXPECTED_COLUMNS if c != "Churn"]
    tmp = _make_csv(columns=cols_without_churn)

    with pytest.raises(ValueError, match="missing columns"):
        load_raw_data(tmp)


# ── Test 6 (Edge case): File not found raises FileNotFoundError ───────────────

def test_nonexistent_file_raises():
    """
    Edge case: passing a path that doesn't exist must raise FileNotFoundError
    with a helpful message, not a pandas crash.
    """
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_raw_data("/nonexistent/path/data.csv")


# ── Test 7 (Edge case): Garbage TotalCharges values raise ValueError ──────────

def test_garbage_total_charges_raises():
    """
    Edge case: TotalCharges containing a non-numeric, non-whitespace string
    (e.g. 'N/A') must raise ValueError — this is not a valid missing-value
    representation in this dataset.

    We force object dtype by mixing numeric strings with 'N/A', ensuring pandas
    cannot infer a purely numeric column and thus reads it as object.
    """
    # Mix of valid strings + one garbage entry → pandas reads as object dtype
    tc_with_garbage = ["1234.56"] * 7041 + [" "] + ["N/A"]  # space is ok, N/A is not
    tmp = _make_csv(n_rows=7043, total_charges_values=tc_with_garbage)

    with pytest.raises(ValueError, match="TotalCharges"):
        load_raw_data(tmp)
