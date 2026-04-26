"""
tests/test_preprocessor.py
Pytest test suite for src/data/preprocessor.py — Day 2.

Run with:
    pytest tests/test_preprocessor.py -v
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import (
    DECAY_LAMBDA,
    audit_missing,
    build_decay_feature,
    build_features,
    build_rfm_features,
    clean_total_charges,
    correlation_audit,
    encode_target,
    quantify_class_imbalance,
)

from src.data.loader import EXPECTED_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Load the real raw dataset once for all tests that need it."""
    if not RAW_CSV.exists():
        pytest.skip("Raw dataset not present — run scripts/download_data.py first.")
    from src.data.loader import load_raw_data
    return load_raw_data(RAW_CSV)


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """
    A tiny synthetic DataFrame that mimics the real schema.
    Used for unit-level tests that don't need the full 7k-row dataset.
    """
    data = {
        "customerID":      ["A001", "A002", "A003", "A004"],
        "gender":          ["Male", "Female", "Male", "Female"],
        "SeniorCitizen":   [0, 1, 0, 0],
        "Partner":         ["Yes", "No", "Yes", "No"],
        "Dependents":      ["No", "No", "Yes", "No"],
        "tenure":          [1, 24, 72, 0],
        "PhoneService":    ["Yes", "Yes", "No", "Yes"],
        "MultipleLines":   ["No", "Yes", "No phone service", "No"],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
        "OnlineSecurity":  ["No", "Yes", "No internet service", "No"],
        "OnlineBackup":    ["Yes", "No", "No internet service", "Yes"],
        "DeviceProtection":["No", "Yes", "No internet service", "No"],
        "TechSupport":     ["No", "No", "No internet service", "No"],
        "StreamingTV":     ["No", "Yes", "No internet service", "No"],
        "StreamingMovies": ["No", "Yes", "No internet service", "No"],
        "Contract":        ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "PaperlessBilling":["Yes", "No", "No", "Yes"],
        "PaymentMethod":   ["Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"],
        "MonthlyCharges":  [29.85, 56.95, 53.85, 42.30],
        "TotalCharges":    ["29.85", "1889.50", " ", "42.30"],   # space = missing
        "Churn":           ["No", "No", "Yes", "Yes"],
    }
    return pd.DataFrame(data)


# ── Test 1: audit_missing correctly counts whitespace in TotalCharges ─────────

def test_audit_missing_detects_whitespace(minimal_df):
    """
    audit_missing() must count the whitespace entry in TotalCharges as missing.
    Asserts: TotalCharges row has total_missing == 1.
    """
    report = audit_missing(minimal_df)
    tc_row = report.loc["TotalCharges"]
    assert tc_row["whitespace_count"] == 1, (
        f"Expected 1 whitespace-missing in TotalCharges, got {tc_row['whitespace_count']}"
    )
    assert tc_row["total_missing"] >= 1


# ── Test 2: quantify_class_imbalance returns correct counts ───────────────────

def test_class_imbalance_counts(minimal_df):
    """
    quantify_class_imbalance() must return class counts summing to len(df),
    and minority_pct must be exactly 50.0 for a balanced 2/2 split.
    Asserts: counts sum == 4, minority_pct == 50.0.
    """
    report = quantify_class_imbalance(minimal_df, target_col="Churn")
    assert sum(report["class_counts"].values()) == len(minimal_df), (
        "Class counts must sum to total number of rows."
    )
    assert report["minority_pct"] == 50.0, (
        f"Expected 50.0% minority for balanced fixture, got {report['minority_pct']}"
    )


# ── Test 3: clean_total_charges imputes whitespace rows correctly ─────────────

def test_clean_total_charges_imputes_whitespace(minimal_df):
    """
    clean_total_charges() must:
      - Convert TotalCharges to float64
      - Impute the whitespace row (tenure=72) with MonthlyCharges × max(tenure,1)
      - Leave non-missing rows unchanged
    Asserts: no NaN remains; imputed value == 53.85 × 72.
    """
    df_clean = clean_total_charges(minimal_df)

    assert df_clean["TotalCharges"].isna().sum() == 0, (
        "No NaN should remain after clean_total_charges()."
    )

    imputed_row = df_clean[df_clean["tenure"] == 72].iloc[0]
    expected = 53.85 * 72
    assert abs(imputed_row["TotalCharges"] - expected) < 0.01, (
        f"Expected imputed TotalCharges ≈ {expected:.2f}, got {imputed_row['TotalCharges']:.2f}"
    )


# ── Test 4: build_rfm_features adds the three RFM columns ─────────────────────

def test_rfm_features_created(minimal_df):
    """
    build_rfm_features() must add rfm_recency, rfm_frequency, rfm_monetary.
    Also validates rfm_frequency counts only active (1-valued) services.
    Asserts: all three columns present; rfm_frequency >= 0 for every row.
    """
    from src.data.preprocessor import (
        encode_binary_yes_no,
        encode_ternary_service_cols,
    )
    df = clean_total_charges(minimal_df)
    df = encode_binary_yes_no(df)
    df = encode_ternary_service_cols(df)
    df = build_rfm_features(df)

    for col in ("rfm_recency", "rfm_frequency", "rfm_monetary"):
        assert col in df.columns, f"Expected column '{col}' not found after build_rfm_features()."

    assert (df["rfm_frequency"] >= 0).all(), "rfm_frequency must be non-negative."
    # Use allclose: rfm_monetary is float32, MonthlyCharges is float64 — tiny rounding delta expected
    assert np.allclose(df["rfm_monetary"].astype("float64"), df["MonthlyCharges"], rtol=1e-4), (
        "rfm_monetary must equal MonthlyCharges within float32 tolerance."
    )


# ── Test 5: build_decay_feature values are in (0, 1] and monotonically correct ─

def test_decay_feature_monotonic(minimal_df):
    """
    build_decay_feature() must produce values in (0, 1].
    Higher tenure → lower decay value (exp is monotonically decreasing in tenure).
    Asserts: tenure=0 → decay=1.0; tenure=72 < tenure=24 decay value.
    """
    df = build_decay_feature(minimal_df, lam=DECAY_LAMBDA)

    assert "tenure_decay" in df.columns

    # Values must be in (0, 1]
    assert (df["tenure_decay"] > 0).all(), "All decay values must be > 0."
    assert (df["tenure_decay"] <= 1.0 + 1e-6).all(), "All decay values must be <= 1.0."

    # tenure=0 → exp(0) = 1.0
    row_zero = df[df["tenure"] == 0].iloc[0]
    assert abs(row_zero["tenure_decay"] - 1.0) < 1e-5, (
        f"Expected decay=1.0 for tenure=0, got {row_zero['tenure_decay']}"
    )

    # Higher tenure → lower decay
    decay_1  = df[df["tenure"] == 1]["tenure_decay"].values[0]
    decay_72 = df[df["tenure"] == 72]["tenure_decay"].values[0]
    assert decay_72 < decay_1, (
        "Longer-tenured customer must have lower decay value."
    )


# ── Test 6 (Edge case): build_features on real data produces parquet ──────────

def test_build_features_produces_parquet(raw_df, tmp_path):
    """
    Edge case / integration: build_features() must:
      - Return a non-empty DataFrame with no raw categorical string columns
      - Write a valid parquet file to the specified output path
      - Contain the 'Churn' column encoded as integer (0/1)
    Asserts: parquet file exists; Churn dtype is int; no object columns remain
             (except those intentionally kept, which should be none post-encoding).
    """
    out = tmp_path / "features_test.parquet"
    df, report = build_features(raw_path=RAW_CSV, output_path=out, save=True)

    # Parquet written
    assert out.exists(), "features.parquet was not created."

    # Churn is integer
    assert df["Churn"].dtype in (np.int8, np.int64, np.int32), (
        f"'Churn' must be integer after encoding, got {df['Churn'].dtype}"
    )

    # No raw object columns should remain (all categoricals encoded)
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    assert len(object_cols) == 0, (
        f"Object-dtype columns remain after encoding: {object_cols}. "
        "All categoricals must be encoded."
    )

    # EDA report keys present
    for key in ("missing", "imbalance", "correlation"):
        assert key in report, f"EDA report missing key: '{key}'"

    # Imbalance report sanity
    assert 20 < report["imbalance"]["minority_pct"] < 35, (
        "Telco dataset churn rate should be between 20–35%."
    )
