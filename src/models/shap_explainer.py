"""
src/models/shap_explainer.py
Day 6 — SHAP Interpretation Layer + Model Evaluation Framework

Components:
  1. SHAP TreeExplainer — global beeswarm, local waterfall, dependence plots
  2. Stratified 5-fold CV evaluation
  3. Temporal train/test split evaluation
  4. Calibration curve + Expected Calibration Error (ECE)
  5. Precision-Recall curve
  6. Confusion matrix at optimal threshold
  7. Serialize SHAP values → data/artifacts/shap_values.parquet
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CV_FOLDS: int = 5
RANDOM_STATE: int = 42
ECE_N_BINS: int = 10          # bins for Expected Calibration Error
FN_COST: float = 5.0
FP_COST: float = 1.0

NON_FEATURE_COLS: list[str] = [
    "Churn", "customerID", "segment", "cluster_id",
    "R_score", "F_score", "M_score", "rfm_score",
]


# ── Helper: extract numeric feature matrix ────────────────────────────────────

def _get_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return (X, feature_names) dropping non-feature and object columns."""
    cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and df[c].dtype != object
    ]
    return df[cols].values.astype(float), cols


def _load_model(model_path: Path):
    """Load a pickled model from disk."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ── 1. SHAP TreeExplainer ─────────────────────────────────────────────────────

def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: list[str],
    max_samples: int = 500,
) -> tuple[np.ndarray, "shap.Explainer"]:
    """
    Compute SHAP values using TreeExplainer on the XGBoost sub-model.

    The calibrated model wraps a pipeline; we unwrap to the raw XGBClassifier
    so TreeExplainer can use the fast tree algorithm (not the slow KernelSHAP).

    Parameters
    ----------
    model        : Fitted CalibratedClassifierCV wrapping SMOTE+XGB pipeline.
    X            : Feature matrix (n_samples, n_features).
    feature_names: Column names matching X columns.
    max_samples  : Cap samples for SHAP computation (speed vs. coverage).

    Returns
    -------
    (shap_values_pos_class, explainer)
    shap_values shape: (n_samples_used, n_features)
    """
    if not _SHAP_AVAILABLE:
        raise RuntimeError("shap not installed. Run: pip install shap")

    # Unwrap CalibratedClassifierCV → pipeline → XGBClassifier
    xgb_model = _unwrap_xgb(model)

    # Sample for speed
    n = min(max_samples, X.shape[0])
    idx = np.random.default_rng(RANDOM_STATE).choice(X.shape[0], size=n, replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_sample)

    # XGBoost binary → single array; multi-output returns list
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # positive class

    return shap_vals, explainer, idx


def _unwrap_xgb(model):
    """
    Unwrap CalibratedClassifierCV → ImbPipeline → XGBClassifier.

    sklearn 1.4 renamed `base_estimator` → `estimator` on _CalibratedClassifier.
    We check both for compatibility.
    """
    base = model  # fallback

    # CalibratedClassifierCV stores calibrated_classifiers_ list after CV fit
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        inner = model.calibrated_classifiers_[0]
        # sklearn ≥ 1.4 uses .estimator; older versions used .base_estimator
        if hasattr(inner, "estimator"):
            base = inner.estimator
        elif hasattr(inner, "base_estimator"):
            base = inner.base_estimator

    # If the calibrator wraps a pipeline directly (cv='prefit' path)
    elif hasattr(model, "estimator"):
        base = model.estimator

    # If base is an ImbPipeline, extract the 'xgb' named step
    if hasattr(base, "named_steps") and "xgb" in base.named_steps:
        return base.named_steps["xgb"]

    return base


# ── 2. Persist SHAP values to Parquet ─────────────────────────────────────────

def save_shap_values(
    shap_values: np.ndarray,
    feature_names: list[str],
    customer_ids: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Serialize SHAP values to parquet for fast dashboard loading.

    Columns: one per feature + optional 'customerID' index.

    Returns
    -------
    Path where the file was saved.
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "artifacts" / "shap_values.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(shap_values, columns=feature_names)
    if customer_ids is not None:
        df.insert(0, "customerID", customer_ids)

    df.to_parquet(output_path, index=False)
    return output_path


# ── 3. Global SHAP Summary (beeswarm data) ────────────────────────────────────

def global_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Compute mean absolute SHAP importance per feature.

    Returns a DataFrame sorted by importance descending,
    top_n rows — used to drive the beeswarm plot in the dashboard.

    Columns: 'feature', 'mean_abs_shap'
    """
    importance = np.abs(shap_values).mean(axis=0)
    df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return df


# ── 4. Local SHAP Waterfall (per-customer) ────────────────────────────────────

def local_shap_waterfall(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: list[str],
    customer_idx: int = 0,
) -> pd.DataFrame:
    """
    Extract SHAP contribution breakdown for a single customer.

    Returns a DataFrame with columns:
      'feature', 'feature_value', 'shap_value'
    sorted by abs(shap_value) descending.

    Parameters
    ----------
    customer_idx : Row index within shap_values / X_sample arrays.
    """
    if customer_idx >= shap_values.shape[0]:
        raise IndexError(
            f"customer_idx={customer_idx} out of range "
            f"(shap_values has {shap_values.shape[0]} rows)"
        )
    sv = shap_values[customer_idx]
    fv = X_sample[customer_idx]
    df = pd.DataFrame({
        "feature":       feature_names,
        "feature_value": fv,
        "shap_value":    sv,
    })
    return df.reindex(df["shap_value"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# ── 5. SHAP Dependence Data (top 3 features) ──────────────────────────────────

def shap_dependence_data(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: list[str],
    top_n: int = 3,
) -> list[pd.DataFrame]:
    """
    Build dependence DataFrames for the top_n most important features.

    Each DataFrame has columns:
      'feature_value', 'shap_value', 'feature_name'

    Returns a list of DataFrames (one per top feature).
    """
    summary = global_shap_summary(shap_values, feature_names, top_n=top_n)
    results = []
    for _, row in summary.iterrows():
        feat = row["feature"]
        idx  = feature_names.index(feat)
        results.append(pd.DataFrame({
            "feature_name":  feat,
            "feature_value": X_sample[:, idx],
            "shap_value":    shap_values[:, idx],
        }))
    return results


# ── 6. Stratified 5-Fold CV Evaluation ───────────────────────────────────────

def stratified_cv_evaluation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = CV_FOLDS,
) -> dict:
    """
    Run stratified k-fold cross-validation and report mean ± std for:
      roc_auc, average_precision (PR-AUC proxy), f1 (minority class).

    Uses the already-fitted model's pipeline structure via cross_validate.

    NOTE: We re-fit clones of the model on each fold — this is purely an
    evaluation pass, not used for the final model weights.

    Returns dict with keys:
      cv_auc_mean, cv_auc_std,
      cv_ap_mean,  cv_ap_std,
      cv_f1_mean,  cv_f1_std
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=["roc_auc", "average_precision", "f1"],
        return_train_score=False,
        n_jobs=-1,
    )

    return {
        "cv_auc_mean": round(float(results["test_roc_auc"].mean()), 4),
        "cv_auc_std":  round(float(results["test_roc_auc"].std()),  4),
        "cv_ap_mean":  round(float(results["test_average_precision"].mean()), 4),
        "cv_ap_std":   round(float(results["test_average_precision"].std()),  4),
        "cv_f1_mean":  round(float(results["test_f1"].mean()), 4),
        "cv_f1_std":   round(float(results["test_f1"].std()),  4),
    }


# ── 7. Temporal Train/Test Split Evaluation ───────────────────────────────────

def temporal_split_evaluation(
    model,
    df: pd.DataFrame,
    tenure_col: str = "tenure",
    target_col: str = "Churn",
    temporal_cutoff_pct: float = 0.80,
) -> dict:
    """
    Simulate temporal leakage check: train on early cohort (low tenure),
    test on recent cohort (high tenure).

    Mimics production: you train on historical customers, score new ones.

    Parameters
    ----------
    temporal_cutoff_pct : Fraction of data (by tenure rank) used as 'train'.

    Returns
    -------
    dict with 'temporal_auc', 'temporal_ap', 'temporal_f1', 'n_train', 'n_test'
    """
    df_sorted = df.sort_values(tenure_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * temporal_cutoff_pct)

    train_df = df_sorted.iloc[:split_idx]
    test_df  = df_sorted.iloc[split_idx:]

    X_train, feat_names = _get_feature_matrix(train_df)
    X_test,  _          = _get_feature_matrix(test_df)
    y_train = train_df[target_col].values.astype(int)
    y_test  = test_df[target_col].values.astype(int)

    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = float(auc(rec, prec))

    return {
        "temporal_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "temporal_ap":  round(ap, 4),
        "temporal_f1":  round(float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "n_train":      int(len(train_df)),
        "n_test":       int(len(test_df)),
    }


# ── 8. Calibration Curve + ECE ────────────────────────────────────────────────

def compute_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = ECE_N_BINS,
) -> dict:
    """
    Compute calibration curve and Expected Calibration Error (ECE).

    ECE = Σ (|bin| / N) × |accuracy(bin) − confidence(bin)|

    A well-calibrated model has ECE ≤ 0.05.

    Returns
    -------
    dict with:
      'fraction_of_positives' (array), 'mean_predicted_value' (array),
      'ece', 'calibration_df' (DataFrame for plotting)
    """
    frac_pos, mean_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )

    # ECE
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        acc  = float(y_true[mask].mean())
        conf = float(y_proba[mask].mean())
        ece += (mask.sum() / n_total) * abs(acc - conf)

    cal_df = pd.DataFrame({
        "mean_predicted_value":  mean_pred,
        "fraction_of_positives": frac_pos,
    })

    return {
        "fraction_of_positives": frac_pos,
        "mean_predicted_value":  mean_pred,
        "ece":                   round(float(ece), 5),
        "calibration_df":        cal_df,
    }


# ── 9. Precision-Recall Curve ─────────────────────────────────────────────────

def compute_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """
    Compute precision-recall curve and PR-AUC.

    Returns
    -------
    dict with 'precision', 'recall', 'thresholds', 'pr_auc', 'pr_df'
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = float(auc(recall, precision))

    pr_df = pd.DataFrame({
        "recall":    recall[:-1],
        "precision": precision[:-1],
        "threshold": thresholds,
    })

    return {
        "precision":  precision,
        "recall":     recall,
        "thresholds": thresholds,
        "pr_auc":     round(pr_auc, 4),
        "pr_df":      pr_df,
    }


# ── 10. Confusion Matrix at Optimal Threshold ─────────────────────────────────

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
) -> dict:
    """
    Find cost-optimal threshold and return full confusion matrix breakdown.

    Returns
    -------
    dict with 'threshold', 'tn', 'fp', 'fn', 'tp',
              'precision', 'recall', 'f1', 'total_cost',
              'cm_df' (2×2 DataFrame for display)
    """
    thresholds = np.unique(y_proba)
    best = {"cost": np.inf, "threshold": 0.5, "tn": 0, "fp": 0, "fn": 0, "tp": 0}

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = fn_cost * fn + fp_cost * fp
        if cost < best["cost"]:
            best = {"cost": cost, "threshold": float(t),
                    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    tn, fp, fn, tp = best["tn"], best["fp"], best["fn"], best["tp"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )

    return {
        "threshold":  round(best["threshold"], 4),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "f1":         round(f1, 4),
        "total_cost": round(best["cost"], 2),
        "cm_df":      cm_df,
    }


# ── 11. Master Evaluation + SHAP Pipeline ─────────────────────────────────────

def run_shap_evaluation(
    features_path: Optional[str | Path] = None,
    model_path:    Optional[str | Path] = None,
    clv_path:      Optional[str | Path] = None,
    shap_output_path: Optional[str | Path] = None,
    max_shap_samples: int = 500,
) -> dict:
    """
    End-to-end Day-6 pipeline:

    1. Load features.parquet + xgb_model.pkl.
    2. Compute SHAP values (TreeExplainer on XGB sub-model).
    3. Save shap_values.parquet.
    4. Global SHAP summary (top 20 features).
    5. SHAP dependence data (top 3 features).
    6. Calibration curve + ECE on held-out 20% test split.
    7. Precision-Recall curve on test split.
    8. Confusion matrix at cost-optimal threshold on test split.
    9. Stratified 5-fold CV metrics.

    Returns
    -------
    dict with all evaluation artefacts and DataFrames.
    """
    # ── Resolve paths ──
    if features_path is None:
        features_path = PROJECT_ROOT / "data" / "processed" / "features.parquet"
    if model_path is None:
        model_path = PROJECT_ROOT / "data" / "artifacts" / "xgb_model.pkl"
    if shap_output_path is None:
        shap_output_path = PROJECT_ROOT / "data" / "artifacts" / "shap_values.parquet"

    features_path    = Path(features_path)
    model_path       = Path(model_path)
    shap_output_path = Path(shap_output_path)

    # ── Load ──
    df    = pd.read_parquet(features_path)
    model = _load_model(model_path)
    X, feature_names = _get_feature_matrix(df)
    y = df["Churn"].values.astype(int)

    # ── Train/test split (stratified, 80/20, same seed as Day 4) ──
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # ── Predictions on test set ──
    y_proba_test = model.predict_proba(X_test)[:, 1]

    # ── SHAP values (on full X for persistence) ──
    shap_values, explainer, sample_idx = compute_shap_values(
        model, X, feature_names, max_samples=max_shap_samples
    )

    # Customer IDs for the SHAP sample
    cust_ids = None
    if "customerID" in df.columns:
        cust_ids = df["customerID"].values[sample_idx]

    shap_path = save_shap_values(shap_values, feature_names, cust_ids, shap_output_path)

    # ── Global summary ──
    global_summary = global_shap_summary(shap_values, feature_names, top_n=20)

    # ── Dependence data (top 3) ──
    X_sample = X[sample_idx]
    dependence_dfs = shap_dependence_data(shap_values, X_sample, feature_names, top_n=3)

    # ── Calibration ──
    calibration = compute_calibration(y_test, y_proba_test)

    # ── PR curve ──
    pr_curve = compute_pr_curve(y_test, y_proba_test)

    # ── Confusion matrix ──
    cm = compute_confusion_matrix(y_test, y_proba_test)

    # ── CV evaluation ──
    cv_metrics = stratified_cv_evaluation(model, X, y)

    return {
        "feature_names":    feature_names,
        "shap_values":      shap_values,
        "shap_path":        shap_path,
        "global_summary":   global_summary,
        "dependence_dfs":   dependence_dfs,
        "calibration":      calibration,
        "pr_curve":         pr_curve,
        "confusion_matrix": cm,
        "cv_metrics":       cv_metrics,
        "X_test":           X_test,
        "y_test":           y_test,
        "y_proba_test":     y_proba_test,
        "X_sample":         X_sample,
        "sample_idx":       sample_idx,
    }
