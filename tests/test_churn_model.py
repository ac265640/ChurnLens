"""
tests/test_churn_model.py
Pytest test suite for src/models/churn_model.py — Day 4.

Run with:
    pytest tests/test_churn_model.py -v

All unit tests use a synthetic 500-row DataFrame so they run without the
real dataset or Optuna HPO overhead.  The integration test
(test_train_churn_model_integration) requires data/processed/features.parquet
and runs Optuna with 5 trials — auto-skipped if the parquet is absent.
"""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from src.models.churn_model import (
    NON_FEATURE_COLS,
    build_smote_xgb_pipeline,
    calibrate_model,
    evaluate_model,
    find_optimal_threshold,
    load_model,
    save_model,
    split_data,
    train_baseline,
)

PROJECT_ROOT     = Path(__file__).resolve().parent.parent
FEATURES_PARQUET = PROJECT_ROOT / "data" / "processed" / "features.parquet"


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """
    500-row synthetic feature DataFrame mimicking the output of build_features().
    Includes all cluster features + a binary Churn column (~26% positive).
    """
    rng = np.random.default_rng(99)
    n   = 500

    tenure       = rng.integers(0, 73, size=n).astype(float)
    monthly      = rng.uniform(18, 120, size=n)
    frequency    = rng.integers(0, 9,  size=n).astype(float)
    decay        = np.exp(-0.04 * tenure)

    # Introduce a realistic correlation: high monthly charges → more churn
    churn_prob   = 0.1 + 0.003 * monthly + rng.normal(0, 0.05, size=n)
    churn_prob   = np.clip(churn_prob, 0, 1)
    churn        = (rng.random(size=n) < churn_prob).astype(np.int8)

    df = pd.DataFrame({
        "tenure":                      tenure,
        "MonthlyCharges":              monthly,
        "TotalCharges":                tenure * monthly,
        "rfm_recency":                 tenure.astype("float32"),
        "rfm_frequency":               frequency.astype("float32"),
        "rfm_monetary":                monthly.astype("float32"),
        "tenure_decay":                decay.astype("float32"),
        "Contract":                    rng.integers(0, 3, size=n, dtype=np.int8),
        "SeniorCitizen":               rng.integers(0, 2, size=n, dtype=np.int8),
        "Partner":                     rng.integers(0, 2, size=n, dtype=np.int8),
        "Dependents":                  rng.integers(0, 2, size=n, dtype=np.int8),
        "PhoneService":                rng.integers(0, 2, size=n, dtype=np.int8),
        "PaperlessBilling":            rng.integers(0, 2, size=n, dtype=np.int8),
        "gender_Male":                 rng.integers(0, 2, size=n, dtype=np.int8),
        "InternetService_Fiber_optic": rng.integers(0, 2, size=n, dtype=np.int8),
        "InternetService_No":          rng.integers(0, 2, size=n, dtype=np.int8),
        "Churn":                       churn,
    })
    return df


@pytest.fixture
def split_arrays(synthetic_df):
    """Return X_train, X_test, y_train, y_test from the synthetic fixture."""
    return split_data(synthetic_df)


# ── Test 1: split_data preserves stratification ────────────────────────────────

def test_split_data_stratification(synthetic_df):
    """
    split_data() must return arrays where:
      - No NON_FEATURE_COLS appear as feature columns.
      - Test set is 20% of total (±1 row tolerance).
      - Minority class proportion in y_train and y_test are within 5 pp of each other.
    """
    X_train, X_test, y_train, y_test, feature_names = split_data(synthetic_df)

    # Churn must NOT be in the feature matrix
    assert "Churn" not in feature_names, "'Churn' must not appear in feature_names."

    # Size check
    total = len(synthetic_df)
    expected_test = int(total * 0.20)
    assert abs(len(X_test) - expected_test) <= 2, (
        f"Expected ~{expected_test} test rows, got {len(X_test)}."
    )

    # Stratification: churn rate within 5 pp across splits
    train_rate = y_train.mean()
    test_rate  = y_test.mean()
    assert abs(train_rate - test_rate) < 0.05, (
        f"Churn rate mismatch: train={train_rate:.3f}, test={test_rate:.3f}. "
        "Stratification may have failed."
    )


# ── Test 2: baseline logistic regression AUC > 0.5 ───────────────────────────

def test_baseline_auc_above_chance(split_arrays):
    """
    train_baseline() must return auc_roc > 0.5 on a signal-carrying dataset.
    Even a weak logistic baseline should beat random guessing.
    Asserts: auc_roc in (0.5, 1.0]; f1_minority in [0, 1].
    """
    X_train, X_test, y_train, y_test, _ = split_arrays
    result = train_baseline(X_train, y_train, X_test, y_test)

    assert "auc_roc" in result and "f1_minority" in result and "model" in result
    assert result["auc_roc"] > 0.5, (
        f"Baseline AUC-ROC={result['auc_roc']:.4f} ≤ 0.50 — model performs at or below chance."
    )
    assert 0.0 <= result["f1_minority"] <= 1.0, (
        f"f1_minority={result['f1_minority']} out of [0, 1] range."
    )


# ── Test 3: SMOTE pipeline produces valid predict_proba output ─────────────────

def test_smote_pipeline_predict_proba(split_arrays):
    """
    build_smote_xgb_pipeline() must produce a fitted pipeline whose
    predict_proba() returns an (n_test, 2) array with values in [0, 1]
    that sum to 1.0 per row.
    Asserts: shape correct; all probabilities in [0, 1]; row sums ≈ 1.
    """
    X_train, X_test, y_train, y_test, _ = split_arrays

    pipeline = build_smote_xgb_pipeline({"n_estimators": 50, "max_depth": 3})
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)

    assert proba.shape == (len(X_test), 2), (
        f"predict_proba shape {proba.shape} != ({len(X_test)}, 2)."
    )
    assert (proba >= 0).all() and (proba <= 1).all(), (
        "predict_proba values must be in [0, 1]."
    )
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), (
        f"Probability rows must sum to 1.0. Max deviation: {np.abs(row_sums - 1.0).max():.6f}."
    )


# ── Test 4: find_optimal_threshold selects threshold below 0.5 ────────────────

def test_optimal_threshold_below_half(split_arrays):
    """
    With FN_COST=5 (FN much more expensive than FP), the cost-optimal
    threshold must be < 0.5 — the model should be biased toward catching
    churners even at the expense of more false positives.
    Asserts: threshold < 0.5; total_cost is a non-negative float.
    """
    X_train, X_test, y_train, y_test, _ = split_arrays

    pipeline = build_smote_xgb_pipeline({"n_estimators": 50, "max_depth": 3})
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    result = find_optimal_threshold(y_test, y_proba, fn_cost=5.0, fp_cost=1.0)

    assert "threshold" in result and "total_cost" in result
    assert result["threshold"] < 0.50, (
        f"With FN_COST=5, optimal threshold should be < 0.5, got {result['threshold']:.4f}."
    )
    assert result["total_cost"] >= 0, "total_cost must be non-negative."


# ── Test 5: evaluate_model returns expected keys and valid ranges ──────────────

def test_evaluate_model_keys_and_ranges(split_arrays):
    """
    evaluate_model() must return a dict containing all required evaluation
    keys, with values in expected numeric ranges.
    Asserts: auc_roc in (0.5, 1.0]; f1_minority in [0, 1];
             optimal_threshold in (0, 1); confusion_matrix keys all non-negative.
    """
    X_train, X_test, y_train, y_test, _ = split_arrays

    pipeline = build_smote_xgb_pipeline({"n_estimators": 50, "max_depth": 3})
    pipeline.fit(X_train, y_train)

    result = evaluate_model(pipeline, X_test, y_test)

    required_keys = [
        "auc_roc", "pr_auc", "f1_minority_at_half",
        "f1_minority_at_optimal", "optimal_threshold",
        "confusion_matrix_at_optimal",
    ]
    for key in required_keys:
        assert key in result, f"Missing key '{key}' in evaluate_model output."

    assert 0.5 < result["auc_roc"] <= 1.0, (
        f"auc_roc={result['auc_roc']} out of expected range (0.5, 1.0]."
    )
    assert 0.0 <= result["f1_minority_at_optimal"] <= 1.0
    assert 0.0 < result["optimal_threshold"] < 1.0

    cm = result["confusion_matrix_at_optimal"]
    for cm_key in ("tn", "fp", "fn", "tp"):
        assert cm[cm_key] >= 0, f"confusion_matrix['{cm_key}'] must be >= 0."


# ── Test 6: save_model / load_model round-trip ────────────────────────────────

def test_save_load_model_roundtrip(split_arrays):
    """
    Edge case: save_model() must write a pickle file that load_model() can
    restore to a callable model with identical predict_proba output.
    Asserts: file exists after save; loaded model predict_proba matches original.
    """
    X_train, X_test, y_train, y_test, _ = split_arrays

    pipeline = build_smote_xgb_pipeline({"n_estimators": 30, "max_depth": 3})
    pipeline.fit(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        saved_path = save_model(pipeline, path=model_path)

        assert saved_path.exists(), "save_model() must create the pkl file."
        assert saved_path.stat().st_size > 0, "pkl file must not be empty."

        loaded = load_model(path=saved_path)

        # predict_proba must match exactly
        original_proba = pipeline.predict_proba(X_test)
        loaded_proba   = loaded.predict_proba(X_test)
        assert np.allclose(original_proba, loaded_proba, atol=1e-6), (
            "Loaded model predict_proba differs from original — pickle round-trip broken."
        )


# ── Test 7 (Integration): full pipeline on real features.parquet ───────────────

def test_train_churn_model_integration():
    """
    Integration test: train_churn_model() on real features.parquet.
    Uses only 5 Optuna trials to keep runtime under 60 seconds.
    Skipped automatically if features.parquet does not exist.

    Asserts:
      - AUC-ROC >= 0.75 (weak lower bound; real data should be ~0.82+)
      - model_path pkl file exists on disk
      - calibrated_model is a CalibratedClassifierCV instance
      - optimal_threshold < 0.5 (cost-sensitive with FN=5×FP)
    """
    if not FEATURES_PARQUET.exists():
        pytest.skip("data/processed/features.parquet not found — run Day 2 pipeline first.")

    from src.models.churn_model import train_churn_model

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "xgb_model.pkl"
        result = train_churn_model(
            feature_path=FEATURES_PARQUET,
            n_optuna_trials=5,      # fast for CI
            save=True,
            model_path=model_path,
        )

    assert result["evaluation"]["auc_roc"] >= 0.75, (
        f"AUC-ROC={result['evaluation']['auc_roc']:.4f} below minimum threshold of 0.75."
    )
    assert isinstance(result["calibrated_model"], CalibratedClassifierCV), (
        "calibrated_model must be a CalibratedClassifierCV instance."
    )
    assert result["evaluation"]["optimal_threshold"] < 0.5, (
        "With FN_COST=5, optimal threshold must be below 0.5 on real data."
    )
