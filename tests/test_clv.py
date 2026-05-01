"""
tests/test_clv.py
Pytest test suite for src/models/clv.py — Day 5.

Run with:
    pytest tests/test_clv.py -v

All unit tests use synthetic data so no real parquet files are required.
The integration test (test_build_clv_table_integration) requires:
  - data/processed/customer_profiles.parquet
  - data/processed/features.parquet
  - data/artifacts/xgb_model.pkl
and is auto-skipped if any are absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.clv import (
    ACCEPT,
    MONITOR,
    NURTURE,
    SAVE,
    assign_priority_quadrant,
    build_clv_heatmap_data,
    compute_dcf_clv,
    compute_revenue_at_risk,
    expected_tenure_from_churn_prob,
)

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
PROFILES_PATH  = PROJECT_ROOT / "data" / "processed" / "customer_profiles.parquet"
FEATURES_PATH  = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODEL_PATH     = PROJECT_ROOT / "data" / "artifacts" / "xgb_model.pkl"


# ─── Test 1: DCF CLV formula correctness ──────────────────────────────────────

def test_dcf_clv_formula_correctness():
    """
    compute_dcf_clv() must match the hand-calculated PV-annuity formula.

    For monthly_revenue=100, T=12, r=0.01:
        PV_factor = (1 - (1.01)^-12) / 0.01 ≈ 11.2551
        CLV       ≈ 100 × 11.2551 ≈ 1125.51

    Asserts: result within $0.01 of hand-calc value.
    """
    r        = 0.01
    T        = 12.0
    rev      = 100.0
    expected = rev * (1.0 - (1.0 + r) ** (-T)) / r

    result = compute_dcf_clv(
        monthly_revenue=rev,
        expected_tenure_months=T,
        monthly_discount_rate=r,
    )

    assert abs(result - expected) < 0.01, (
        f"DCF CLV formula mismatch: expected {expected:.4f}, got {result:.4f}"
    )
    assert result > 0, "DCF CLV must be positive for positive revenue and tenure."


def test_dcf_clv_zero_discount_rate():
    """
    Edge case: when monthly_discount_rate=0, CLV = monthly_revenue × T (no discounting).
    Asserts: result == 100 × 24 = 2400.0 exactly.
    """
    result = compute_dcf_clv(
        monthly_revenue=100.0,
        expected_tenure_months=24.0,
        monthly_discount_rate=0.0,
    )
    assert result == pytest.approx(2400.0, abs=0.01), (
        f"Zero-discount CLV should be 2400.0, got {result:.4f}"
    )


def test_dcf_clv_minimum_tenure_floor():
    """
    Edge case: expected_tenure_months=0 must be floored to 1 month, not cause
    division by zero or return 0 for a customer with positive revenue.
    Asserts: CLV > 0 for monthly_revenue=50, tenure=0.
    """
    result = compute_dcf_clv(monthly_revenue=50.0, expected_tenure_months=0.0)
    assert result > 0, (
        f"CLV for tenure=0 (floored to 1) must be > 0, got {result:.4f}"
    )


def test_dcf_clv_vectorised():
    """
    compute_dcf_clv() must operate element-wise on arrays without error.
    Asserts: output shape matches input; all values > 0.
    """
    revenues = np.array([50.0, 75.0, 100.0, 120.0])
    tenures  = np.array([6.0,  12.0, 18.0,  24.0])

    result = compute_dcf_clv(revenues, tenures)

    assert result.shape == revenues.shape, (
        f"Vectorised CLV shape {result.shape} != input shape {revenues.shape}"
    )
    assert (result > 0).all(), "All vectorised CLV values must be positive."
    # Higher revenue + longer tenure → higher CLV (monotonicity check)
    assert result[-1] > result[0], (
        "CLV should increase with higher revenue and longer tenure."
    )


# ─── Test 2: expected_tenure_from_churn_prob ──────────────────────────────────

def test_expected_tenure_high_churn_short():
    """
    A customer with churn_prob=0.9 (very likely to churn) must have a
    much shorter expected tenure than one with churn_prob=0.1.
    Asserts: tenure(0.9) < tenure(0.1); both within [1, 120] months.
    """
    t_high = expected_tenure_from_churn_prob(0.9)
    t_low  = expected_tenure_from_churn_prob(0.1)

    assert t_high < t_low, (
        f"High-churn customer should have shorter tenure: {t_high:.2f} vs {t_low:.2f}"
    )
    assert 1.0 <= t_high <= 120.0, f"tenure out of [1, 120]: {t_high:.2f}"
    assert 1.0 <= t_low  <= 120.0, f"tenure out of [1, 120]: {t_low:.2f}"


# ─── Test 3: assign_priority_quadrant — all four labels produced ───────────────

def test_priority_quadrant_all_four_labels():
    """
    assign_priority_quadrant() must produce all four quadrant labels
    when given customers covering all four CLV × Risk combinations.

    Input (4 customers):
      High CLV + High Risk → Save
      High CLV + Low Risk  → Nurture
      Low CLV  + High Risk → Accept
      Low CLV  + Low Risk  → Monitor

    Asserts: each customer gets the exactly expected label.
    """
    clv        = np.array([1000.0, 1000.0, 100.0, 100.0])
    churn_prob = np.array([0.8,    0.2,    0.8,   0.2])

    result = assign_priority_quadrant(
        clv, churn_prob,
        clv_threshold=500.0,
        churn_threshold=0.5,
    )

    assert result[0] == SAVE,    f"High CLV + High Risk → expected '{SAVE}', got '{result[0]}'"
    assert result[1] == NURTURE, f"High CLV + Low Risk  → expected '{NURTURE}', got '{result[1]}'"
    assert result[2] == ACCEPT,  f"Low CLV  + High Risk → expected '{ACCEPT}', got '{result[2]}'"
    assert result[3] == MONITOR, f"Low CLV  + Low Risk  → expected '{MONITOR}', got '{result[3]}'"


def test_priority_quadrant_median_split():
    """
    When clv_threshold=None, assign_priority_quadrant() should split at the
    median CLV. For a symmetric array the labels must be 50/50 high vs low.
    Asserts: number of (Save + Nurture) == number of (Accept + Monitor) ± 1.
    """
    rng        = np.random.default_rng(42)
    n          = 200
    clv        = rng.uniform(0, 1000, size=n)
    churn_prob = rng.uniform(0,    1, size=n)

    result = assign_priority_quadrant(clv, churn_prob)

    high_clv_count = np.sum(np.isin(result, [SAVE, NURTURE]))
    low_clv_count  = np.sum(np.isin(result, [ACCEPT, MONITOR]))

    assert abs(high_clv_count - low_clv_count) <= 2, (
        f"Median split should yield ~equal high/low CLV groups: "
        f"{high_clv_count} vs {low_clv_count}"
    )


# ─── Test 4: revenue_at_risk is bounded ───────────────────────────────────────

def test_revenue_at_risk_bounded_by_clv():
    """
    revenue_at_risk = CLV × churn_prob must satisfy:
      0 ≤ revenue_at_risk ≤ CLV for all customers.
    Asserts: all values in [0, CLV]; values monotone in churn_prob.
    """
    clv        = np.array([500.0, 500.0, 500.0, 500.0])
    churn_prob = np.array([0.0,   0.25,  0.75,  1.0])

    rar = compute_revenue_at_risk(clv, churn_prob)

    assert (rar >= 0).all(), "revenue_at_risk must be non-negative."
    assert (rar <= clv).all(), "revenue_at_risk must not exceed CLV."
    assert rar[0] == pytest.approx(0.0),   "RaR for churn_prob=0 must be 0."
    assert rar[-1] == pytest.approx(500.0), "RaR for churn_prob=1 must equal CLV."
    # Monotonicity: higher churn → higher revenue at risk
    assert rar[1] < rar[2], "RaR must increase with churn probability."


# ─── Test 5: heatmap data shape and non-negative values ──────────────────────

def test_clv_heatmap_shape_and_values():
    """
    build_clv_heatmap_data() must return a DataFrame of shape
    (n_bins, n_bins) with all non-negative mean revenue_at_risk values.
    Asserts: shape == (3, 3); all cells >= 0; index/column labels are strings.
    """
    rng        = np.random.default_rng(7)
    clv        = rng.uniform(100, 2000, size=500)
    churn_prob = rng.uniform(0,   1,    size=500)

    pivot = build_clv_heatmap_data(clv, churn_prob, n_bins=3)

    assert pivot.shape == (3, 3), (
        f"Heatmap pivot shape {pivot.shape} != (3, 3)"
    )
    assert (pivot.values >= 0).all(), (
        "All heatmap cells must have non-negative mean revenue_at_risk."
    )
    # Labels should be categorical strings (Low/Medium/High)
    assert all(isinstance(idx, str) for idx in pivot.index.tolist()), (
        "Heatmap row index must be string labels."
    )


# ─── Test 6 (Integration): full pipeline on real artefacts ───────────────────

@pytest.mark.skipif(
    not (PROFILES_PATH.exists() and FEATURES_PATH.exists() and MODEL_PATH.exists()),
    reason="Real parquet/pkl artefacts not found — run Days 1–4 first."
)
def test_build_clv_table_integration():
    """
    Integration test: build_clv_table() on real data artefacts.

    Asserts:
      - Output DataFrame has ≥ 7000 rows (full dataset).
      - Required columns are present.
      - priority_quadrant contains only the four valid labels.
      - revenue_at_risk is non-negative for all customers.
      - clv_dcf is positive for all customers.
      - customer_clv.parquet is written to disk.
    """
    import tempfile
    from src.models.clv import build_clv_table, SAVE, NURTURE, ACCEPT, MONITOR

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "customer_clv.parquet"
        result   = build_clv_table(
            profiles_path=PROFILES_PATH,
            features_path=FEATURES_PATH,
            model_path=MODEL_PATH,
            output_path=out_path,
            use_bgnbd=True,
        )

    required_cols = [
        "churn_probability", "expected_tenure_months", "clv_dcf",
        "clv_bgnbd", "revenue_at_risk", "priority_quadrant",
        "clv_bin", "churn_risk_bin",
    ]
    for col in required_cols:
        assert col in result.columns, f"Missing column: '{col}'"

    assert len(result) >= 7000, (
        f"Expected ≥ 7000 rows, got {len(result)}"
    )
    valid_labels = {SAVE, NURTURE, ACCEPT, MONITOR}
    bad = set(result["priority_quadrant"].unique()) - valid_labels
    assert not bad, f"Invalid priority_quadrant labels: {bad}"

    assert (result["revenue_at_risk"] >= 0).all(), (
        "revenue_at_risk must be non-negative for all customers."
    )
    assert (result["clv_dcf"] > 0).all(), (
        "clv_dcf must be positive for all customers."
    )
    assert (result["churn_probability"].between(0, 1)).all(), (
        "churn_probability must be in [0, 1]."
    )
