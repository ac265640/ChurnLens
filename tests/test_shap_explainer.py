"""
tests/test_shap_explainer.py
Day 6 — SHAP Interpretation + Model Evaluation Test Suite

12 tests covering:
  1.  SHAP values shape is correct
  2.  SHAP values sum to non-zero (not all zeroes)
  3.  global_shap_summary returns sorted DataFrame, correct columns
  4.  local_shap_waterfall returns correct shape and sorted by abs value
  5.  shap_dependence_data returns 3 DataFrames with correct columns
  6.  compute_calibration returns ECE value and DataFrame
  7.  ECE is a non-negative float
  8.  compute_pr_curve returns pr_auc in [0, 1] and DataFrame
  9.  compute_confusion_matrix returns tn+fp+fn+tp == n_test
  10. compute_confusion_matrix threshold is in [0, 1]
  11. save_shap_values persists parquet and columns match feature names
  12. stratified_cv_evaluation returns expected keys with valid ranges
"""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# ── Module under test ──────────────────────────────────────────────────────────
from src.models.shap_explainer import (
    compute_calibration,
    compute_confusion_matrix,
    compute_pr_curve,
    compute_shap_values,
    global_shap_summary,
    local_shap_waterfall,
    save_shap_values,
    shap_dependence_data,
    stratified_cv_evaluation,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_data():
    """Binary classification dataset with 10 features, 26% positive rate."""
    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=6,
        n_redundant=2, weights=[0.74, 0.26],
        random_state=42, flip_y=0.02,
    )
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    return X, y, feature_names


@pytest.fixture(scope="module")
def fitted_lr(synthetic_data):
    """A simple logistic regression (not XGB) — used for non-SHAP tests."""
    X, y, _ = synthetic_data
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X[:320], y[:320])
    return lr


@pytest.fixture(scope="module")
def fitted_xgb_model(synthetic_data):
    """
    Minimal calibrated XGBoost pipeline matching Day-4 structure,
    so TreeExplainer can unwrap it.
    """
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from sklearn.calibration import CalibratedClassifierCV
    from xgboost import XGBClassifier

    X, y, _ = synthetic_data
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("xgb",   XGBClassifier(n_estimators=50, max_depth=3,
                                 random_state=42, eval_metric="logloss")),
    ])
    calibrated = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=3)
    calibrated.fit(X[:320], y[:320])
    return calibrated


@pytest.fixture(scope="module")
def shap_results(fitted_xgb_model, synthetic_data):
    """Pre-computed SHAP values for reuse across tests."""
    X, y, feature_names = synthetic_data
    shap_vals, explainer, sample_idx = compute_shap_values(
        fitted_xgb_model, X, feature_names, max_samples=200
    )
    return shap_vals, explainer, sample_idx, X[sample_idx], feature_names


@pytest.fixture(scope="module")
def test_predictions(fitted_xgb_model, synthetic_data):
    """Held-out test probabilities and labels."""
    X, y, _ = synthetic_data
    X_test, y_test = X[320:], y[320:]
    y_proba = fitted_xgb_model.predict_proba(X_test)[:, 1]
    return y_test, y_proba


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestShapValues:
    """Tests 1–5: SHAP value computation."""

    def test_shap_values_shape(self, shap_results, synthetic_data):
        """SHAP output shape must be (n_samples_used, n_features)."""
        shap_vals, _, sample_idx, X_sample, feature_names = shap_results
        assert shap_vals.shape[0] == len(sample_idx), (
            f"Expected {len(sample_idx)} rows, got {shap_vals.shape[0]}"
        )
        assert shap_vals.shape[1] == len(feature_names), (
            f"Expected {len(feature_names)} cols, got {shap_vals.shape[1]}"
        )

    def test_shap_values_nonzero(self, shap_results):
        """Not all SHAP values should be zero — model must have learned."""
        shap_vals, *_ = shap_results
        assert np.abs(shap_vals).sum() > 0, "All SHAP values are zero — model did not learn"

    def test_global_summary_columns_and_order(self, shap_results):
        """global_shap_summary: correct columns, sorted descending."""
        shap_vals, _, _, _, feature_names = shap_results
        summary = global_shap_summary(shap_vals, feature_names, top_n=5)
        assert list(summary.columns) == ["feature", "mean_abs_shap"], (
            f"Unexpected columns: {list(summary.columns)}"
        )
        assert len(summary) <= 5
        vals = summary["mean_abs_shap"].values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)), (
            "global_shap_summary is not sorted descending"
        )

    def test_local_waterfall_shape_and_sort(self, shap_results):
        """local_shap_waterfall: correct row count, sorted by abs SHAP."""
        shap_vals, _, _, X_sample, feature_names = shap_results
        wf = local_shap_waterfall(shap_vals, X_sample, feature_names, customer_idx=0)
        assert len(wf) == len(feature_names), (
            f"Expected {len(feature_names)} rows, got {len(wf)}"
        )
        assert list(wf.columns) == ["feature", "feature_value", "shap_value"]
        abs_vals = wf["shap_value"].abs().values
        assert all(abs_vals[i] >= abs_vals[i + 1] for i in range(len(abs_vals) - 1)), (
            "local_shap_waterfall not sorted by abs shap_value"
        )

    def test_dependence_data_structure(self, shap_results):
        """shap_dependence_data: returns 3 DataFrames with correct columns."""
        shap_vals, _, _, X_sample, feature_names = shap_results
        dfs = shap_dependence_data(shap_vals, X_sample, feature_names, top_n=3)
        assert len(dfs) == 3, f"Expected 3 DataFrames, got {len(dfs)}"
        for df in dfs:
            assert "feature_name"  in df.columns
            assert "feature_value" in df.columns
            assert "shap_value"    in df.columns
            assert len(df) > 0


class TestCalibration:
    """Tests 6–7: calibration curve and ECE."""

    def test_calibration_returns_ece_and_df(self, test_predictions):
        """compute_calibration must return 'ece' and 'calibration_df'."""
        y_test, y_proba = test_predictions
        result = compute_calibration(y_test, y_proba)
        assert "ece" in result
        assert "calibration_df" in result
        assert isinstance(result["calibration_df"], pd.DataFrame)
        assert "mean_predicted_value"  in result["calibration_df"].columns
        assert "fraction_of_positives" in result["calibration_df"].columns

    def test_ece_is_nonnegative_float(self, test_predictions):
        """ECE must be a float in [0, 1]."""
        y_test, y_proba = test_predictions
        result = compute_calibration(y_test, y_proba)
        ece = result["ece"]
        assert isinstance(ece, float), f"ECE type: {type(ece)}"
        assert 0.0 <= ece <= 1.0, f"ECE out of range: {ece}"


class TestPRCurve:
    """Test 8: precision-recall curve."""

    def test_pr_curve_auc_and_df(self, test_predictions):
        """pr_auc must be in [0, 1]; pr_df must have recall, precision columns."""
        y_test, y_proba = test_predictions
        result = compute_pr_curve(y_test, y_proba)
        assert 0.0 <= result["pr_auc"] <= 1.0, f"PR-AUC out of range: {result['pr_auc']}"
        assert "recall"    in result["pr_df"].columns
        assert "precision" in result["pr_df"].columns
        assert "threshold" in result["pr_df"].columns


class TestConfusionMatrix:
    """Tests 9–10: confusion matrix at optimal threshold."""

    def test_cm_totals_equal_n_test(self, test_predictions):
        """tn + fp + fn + tp must equal the number of test samples."""
        y_test, y_proba = test_predictions
        result = compute_confusion_matrix(y_test, y_proba)
        total = result["tn"] + result["fp"] + result["fn"] + result["tp"]
        assert total == len(y_test), (
            f"CM totals ({total}) != n_test ({len(y_test)})"
        )

    def test_threshold_in_unit_interval(self, test_predictions):
        """Optimal threshold must be a probability in [0, 1]."""
        y_test, y_proba = test_predictions
        result = compute_confusion_matrix(y_test, y_proba)
        t = result["threshold"]
        assert 0.0 <= t <= 1.0, f"Threshold {t} out of [0, 1]"


class TestPersistence:
    """Test 11: SHAP parquet serialization."""

    def test_save_shap_values_creates_parquet(self, shap_results):
        """save_shap_values must create a readable parquet with correct columns."""
        shap_vals, _, _, _, feature_names = shap_results
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "shap_test.parquet"
            returned_path = save_shap_values(shap_vals, feature_names, output_path=out_path)
            assert returned_path.exists(), "Parquet file was not created"
            df = pd.read_parquet(returned_path)
            assert list(df.columns) == feature_names, (
                f"Column mismatch:\n  got:      {list(df.columns)}\n  expected: {feature_names}"
            )
            assert len(df) == shap_vals.shape[0]


class TestCVEvaluation:
    """Test 12: stratified 5-fold CV."""

    def test_cv_evaluation_keys_and_ranges(self, fitted_xgb_model, synthetic_data):
        """stratified_cv_evaluation must return 6 expected keys, all in valid ranges."""
        X, y, _ = synthetic_data
        result = stratified_cv_evaluation(fitted_xgb_model, X, y, n_splits=3)
        expected_keys = [
            "cv_auc_mean", "cv_auc_std",
            "cv_ap_mean",  "cv_ap_std",
            "cv_f1_mean",  "cv_f1_std",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        assert 0.5 <= result["cv_auc_mean"] <= 1.0, (
            f"cv_auc_mean={result['cv_auc_mean']} not in [0.5, 1.0]"
        )
        assert result["cv_auc_std"] >= 0.0
        assert 0.0 <= result["cv_f1_mean"] <= 1.0
