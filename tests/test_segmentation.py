"""
tests/test_segmentation.py
Pytest test suite for src/features/segmentation.py — Day 3.

Run with:
    pytest tests/test_segmentation.py -v

All unit tests use a synthetic 200-row DataFrame so they run without the
real dataset.  The integration test (test_run_segmentation_integration)
requires data/processed/features.parquet and is auto-skipped if absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.segmentation import (
    CLUSTER_FEATURES,
    SEGMENT_NAMES,
    assign_segment_names,
    bootstrap_stability,
    build_segment_profile,
    fit_kmeans,
    run_segmentation,
    score_rfm_quartiles,
    select_optimal_k,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PARQUET = PROJECT_ROOT / "data" / "processed" / "features.parquet"


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """
    200-row synthetic DataFrame that mimics the processed feature matrix
    produced by Day 2's build_features().  Contains all CLUSTER_FEATURES
    plus auxiliary columns needed by the pipeline.
    """
    rng = np.random.default_rng(42)
    n = 200

    tenure         = rng.integers(0, 73, size=n).astype(float)
    monthly        = rng.uniform(18, 120, size=n)
    frequency      = rng.integers(0, 9, size=n).astype(float)
    tenure_decay   = np.exp(-0.04 * tenure).astype(float)
    churn          = rng.integers(0, 2, size=n, dtype=np.int8)

    df = pd.DataFrame({
        "tenure":         tenure,
        "MonthlyCharges": monthly,
        "TotalCharges":   tenure * monthly,
        "rfm_recency":    tenure.astype("float32"),
        "rfm_frequency":  frequency.astype("float32"),
        "rfm_monetary":   monthly.astype("float32"),
        "tenure_decay":   tenure_decay.astype("float32"),
        "Churn":          churn,
        # Dummy encoded columns (required so no object cols remain)
        "Contract":       rng.integers(0, 3, size=n, dtype=np.int8),
        "SeniorCitizen":  rng.integers(0, 2, size=n, dtype=np.int8),
    })
    return df


# ── Test 1: score_rfm_quartiles adds correct columns with valid ranges ─────────

def test_rfm_score_columns_and_range(synthetic_df):
    """
    score_rfm_quartiles() must add R_score, F_score, M_score, rfm_score.
    Each individual score must be in [1, 4].
    Composite rfm_score must be in [3, 12].
    Asserts: all four columns present; value ranges satisfied across 200 rows.
    """
    df = score_rfm_quartiles(synthetic_df)

    for col in ("R_score", "F_score", "M_score", "rfm_score"):
        assert col in df.columns, f"Expected column '{col}' missing after score_rfm_quartiles()."

    for col in ("R_score", "F_score", "M_score"):
        assert df[col].between(1, 4).all(), (
            f"{col} values must be in [1, 4]. "
            f"Found min={df[col].min()}, max={df[col].max()}."
        )

    assert df["rfm_score"].between(3, 12).all(), (
        f"rfm_score must be in [3, 12]. "
        f"Found min={df['rfm_score'].min()}, max={df['rfm_score'].max()}."
    )


# ── Test 2: select_optimal_k returns valid recommended_k ──────────────────────

def test_select_optimal_k_returns_valid_k(synthetic_df):
    """
    select_optimal_k() must return a recommended_k in [K_MIN, K_MAX],
    and both 'inertias' and 'silhouettes' dicts must be non-empty.
    Asserts: recommended_k is integer in [2, 8]; inertias has >= 3 entries.
    """
    from sklearn.preprocessing import StandardScaler
    available = [c for c in CLUSTER_FEATURES if c in synthetic_df.columns]
    X_scaled = StandardScaler().fit_transform(synthetic_df[available].values.astype(float))

    result = select_optimal_k(X_scaled, k_min=2, k_max=6)

    assert "recommended_k" in result
    assert isinstance(result["recommended_k"], int)
    assert 2 <= result["recommended_k"] <= 6, (
        f"recommended_k={result['recommended_k']} out of expected range [2, 6]."
    )
    assert len(result["inertias"]) >= 3, "Expected at least 3 inertia entries for k=2..6."
    assert len(result["silhouettes"]) >= 2, "Expected at least 2 silhouette entries."


# ── Test 3: fit_kmeans adds cluster_id and assigns correct number of clusters ──

def test_fit_kmeans_cluster_count(synthetic_df):
    """
    fit_kmeans() must add a 'cluster_id' column with exactly k distinct values.
    Asserts: cluster_id present; exactly k=4 unique values for 200-row input.
    """
    df_out, km, scaler, X_scaled = fit_kmeans(synthetic_df, k=4)

    assert "cluster_id" in df_out.columns, "'cluster_id' column missing after fit_kmeans()."
    unique_clusters = df_out["cluster_id"].nunique()
    assert unique_clusters == 4, (
        f"Expected 4 distinct cluster IDs, got {unique_clusters}."
    )
    assert len(df_out) == len(synthetic_df), "fit_kmeans() must not change row count."


# ── Test 4: bootstrap_stability ARI is in [0, 1] and stability flag correct ───

def test_bootstrap_stability_ari_bounds(synthetic_df):
    """
    bootstrap_stability() must return ari_mean in [0.0, 1.0].
    For a clear 4-cluster synthetic dataset, ARI should be >= 0.5.
    Asserts: ari_mean ∈ [0, 1]; ari_std >= 0; 'stable' is bool.
    """
    from sklearn.preprocessing import StandardScaler
    available = [c for c in CLUSTER_FEATURES if c in synthetic_df.columns]
    X_scaled = StandardScaler().fit_transform(synthetic_df[available].values.astype(float))

    result = bootstrap_stability(X_scaled, k=4, n_runs=5, seed_start=0)

    assert 0.0 <= result["ari_mean"] <= 1.0, (
        f"ari_mean must be in [0, 1], got {result['ari_mean']}."
    )
    assert result["ari_std"] >= 0, "ari_std must be non-negative."
    assert isinstance(result["stable"], bool), "'stable' must be a boolean."
    assert result["n_runs"] == 5


# ── Test 5: assign_segment_names maps every row to a known segment label ───────

def test_segment_names_coverage(synthetic_df):
    """
    After fit_kmeans + assign_segment_names, every row must have a non-null
    segment label drawn from SEGMENT_NAMES (or extended names for k > 4).
    Asserts: 'segment' column present; no NaN; all values in allowed label set.
    """
    df_out, km, scaler, _ = fit_kmeans(synthetic_df, k=4)
    df_named = assign_segment_names(df_out, km, scaler=scaler)

    assert "segment" in df_named.columns, "'segment' column missing after assign_segment_names()."
    assert df_named["segment"].isna().sum() == 0, "No segment label should be NaN."

    allowed = set(SEGMENT_NAMES[:4])
    actual  = set(df_named["segment"].unique())
    assert actual.issubset(allowed), (
        f"Unexpected segment labels found: {actual - allowed}. "
        f"All labels must come from {allowed}."
    )


# ── Test 6: build_segment_profile includes churn_rate and sums to 100% ─────────

def test_segment_profile_completeness(synthetic_df):
    """
    Edge case: build_segment_profile() must include churn_rate_pct when Churn
    column is present, and pct_of_total must sum to 100% (within rounding).
    Asserts: churn_rate_pct present; sum(pct_of_total) ≈ 100; n_customers > 0.
    """
    df_out, km, scaler, _ = fit_kmeans(synthetic_df, k=4)
    df_named = assign_segment_names(df_out, km, scaler=scaler)
    profile = build_segment_profile(df_named)

    # All segments have at least one customer
    assert (profile["n_customers"] > 0).all(), "Every segment must have at least one customer."

    # Percentages sum to ~100
    pct_sum = profile["pct_of_total"].sum()
    assert abs(pct_sum - 100.0) < 0.5, (
        f"pct_of_total should sum to ~100, got {pct_sum:.1f}."
    )

    # Churn rate present (Churn col is in synthetic_df)
    assert "churn_rate_pct" in profile.columns, (
        "churn_rate_pct must be in profile when Churn column is present."
    )
    assert profile["churn_rate_pct"].between(0, 100).all(), (
        "churn_rate_pct values must be between 0 and 100."
    )


# ── Test 7 (Integration): run_segmentation on real features.parquet ───────────

def test_run_segmentation_integration():
    """
    Integration test: run_segmentation() on the real processed feature file.
    Skipped automatically if features.parquet does not exist.
    Asserts: profiles_df has 'segment' column; stability['ari_mean'] >= 0;
             segment_summary has expected columns; k_used in [2, 8].
    """
    if not FEATURES_PARQUET.exists():
        pytest.skip("data/processed/features.parquet not found — run Day 2 pipeline first.")

    result = run_segmentation(feature_path=FEATURES_PARQUET, save=False)

    df = result["profiles_df"]
    assert "segment" in df.columns, "profiles_df must contain 'segment' column."
    assert len(df) > 0, "profiles_df must not be empty."

    summary = result["segment_summary"]
    assert "n_customers" in summary.columns
    assert "mean_rfm_score" in summary.columns

    assert result["stability"]["ari_mean"] >= 0
    assert 2 <= result["k_used"] <= 8, (
        f"k_used={result['k_used']} out of expected range [2, 8]."
    )
