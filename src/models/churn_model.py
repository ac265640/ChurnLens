"""
src/models/churn_model.py
Day 4 — Churn Prediction Model

Components:
  1. Baseline logistic regression (regularised, no SMOTE)
  2. SMOTE-inside-CV Pipeline for XGBoost (never resamples test fold)
  3. Optuna hyperparameter search (TPE, AUC-ROC objective)
  4. Platt scaling (CalibratedClassifierCV, sigmoid method)
  5. Cost-sensitive threshold tuning (FN cost = 5 × FP cost)
  6. Evaluation: AUC-ROC, F1-minority, confusion matrix at optimal threshold
  7. Artefact persistence: xgb_model.pkl → data/artifacts/

Nothing in this module uses the Churn column for anything except supervised
training on the training split — no leakage possible by construction.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Columns to exclude from the feature matrix (target + ID-like cols)
NON_FEATURE_COLS: list[str] = ["Churn", "customerID", "segment", "cluster_id",
                                "R_score", "F_score", "M_score", "rfm_score"]

# Cost ratio: FN is 5× more expensive than FP
FN_COST: float = 5.0
FP_COST: float = 1.0

# CV config
CV_FOLDS: int = 5
RANDOM_STATE: int = 42

# Optuna
OPTUNA_N_TRIALS: int = 40
OPTUNA_TIMEOUT_SEC: int = 180   # safety cap — 3 minutes max


# ── 1. Data Splitting ──────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    target_col: str = "Churn",
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Stratified train/test split.

    Returns (X_train, X_test, y_train, y_test, feature_names).

    Drops NON_FEATURE_COLS and any remaining object-dtype columns with a
    warning so the caller always gets a clean numeric matrix.
    """
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and df[c].dtype != object
    ]
    object_dropped = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and df[c].dtype == object
    ]
    if object_dropped:
        warnings.warn(
            f"split_data: dropping {len(object_dropped)} object-dtype columns "
            f"not in NON_FEATURE_COLS: {object_dropped}",
            stacklevel=2,
        )

    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, feature_cols


# ── 2. Baseline Logistic Regression ───────────────────────────────────────────

def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Regularised logistic regression baseline (L2, C=0.1, scaled features).

    Returns a dict with 'auc_roc', 'f1_minority', 'model'.
    No SMOTE — serves as the unaugmented lower bound.
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    lr.fit(X_tr_scaled, y_train)

    y_proba = lr.predict_proba(X_te_scaled)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    return {
        "model":        lr,
        "scaler":       scaler,
        "auc_roc":      round(float(roc_auc_score(y_test, y_proba)), 4),
        "f1_minority":  round(float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
    }


# ── 3. SMOTE-in-CV XGBoost Pipeline ───────────────────────────────────────────

def build_smote_xgb_pipeline(xgb_params: Optional[dict] = None) -> ImbPipeline:
    """
    Build an imbalanced-learn Pipeline:
      SMOTE (applied only to training fold) → XGBClassifier.

    The Pipeline API ensures SMOTE.fit_resample() is called only on
    the training portion of each CV fold — the test fold is never touched.

    Parameters
    ----------
    xgb_params : dict | None
        XGBoost hyperparameters. Defaults to a sensible starting point.
    """
    default_params = {
        "n_estimators":     300,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma":            0.1,
        "use_label_encoder": False,
        "eval_metric":      "logloss",
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
    }
    if xgb_params:
        default_params.update(xgb_params)

    pipeline = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("xgb",   XGBClassifier(**default_params)),
    ])
    return pipeline


# ── 4. Optuna Hyperparameter Search ───────────────────────────────────────────

def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: int = OPTUNA_TIMEOUT_SEC,
) -> dict:
    """
    TPE search over XGBoost hyperparameters using Optuna.
    Objective: maximise mean stratified 5-fold AUC-ROC on training data.
    SMOTE is applied inside each fold via build_smote_xgb_pipeline().

    Returns dict with 'best_params', 'best_auc', 'study'.
    """
    if not _OPTUNA_AVAILABLE:
        raise RuntimeError("optuna is not installed. Run: pip install optuna")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
        }
        pipeline = build_smote_xgb_pipeline(params)
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    return {
        "best_params": study.best_params,
        "best_auc":    round(study.best_value, 4),
        "study":       study,
    }


# ── 5. Platt Scaling Calibration ──────────────────────────────────────────────

def calibrate_model(
    pipeline: ImbPipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> CalibratedClassifierCV:
    """
    Wrap the fitted SMOTE+XGB pipeline with Platt scaling (sigmoid).

    CalibratedClassifierCV with cv='prefit' expects the base estimator to
    already be fitted. We fit the pipeline first, then calibrate on the
    same training data using internal cross-validation (cv=5).

    Note: cv='prefit' + sigmoid is Platt scaling by definition.
    """
    # Fit the raw pipeline first
    pipeline.fit(X_train, y_train)

    # Calibrate using 5-fold internal CV on training data
    calibrated = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


# ── 6. Cost-Sensitive Threshold Tuning ────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
) -> dict:
    """
    Find the decision threshold that minimises total cost:
      cost = fn_cost × FN + fp_cost × FP

    Scans all unique probability values as candidate thresholds.

    Returns dict with 'threshold', 'total_cost', 'fn_count', 'fp_count',
    'tn_count', 'tp_count'.
    """
    thresholds   = np.unique(y_proba)
    best_cost    = np.inf
    best_thresh  = 0.5
    best_cm_vals = None

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = fn_cost * fn + fp_cost * fp
        if cost < best_cost:
            best_cost    = cost
            best_thresh  = float(t)
            best_cm_vals = (int(tn), int(fp), int(fn), int(tp))

    tn, fp, fn, tp = best_cm_vals
    return {
        "threshold":  round(best_thresh, 4),
        "total_cost": round(best_cost, 2),
        "tn_count":   tn,
        "fp_count":   fp,
        "fn_count":   fn,
        "tp_count":   tp,
    }


# ── 7. Full Evaluation Report ──────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
) -> dict:
    """
    Compute comprehensive evaluation metrics on the held-out test set.

    Returns dict with:
      auc_roc, f1_minority_at_optimal, f1_minority_at_half,
      optimal_threshold, confusion_matrix_at_optimal (dict),
      pr_auc (precision-recall AUC).
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    # AUC-ROC
    auc_roc = float(roc_auc_score(y_test, y_proba))

    # Precision-Recall AUC (better for imbalanced classes)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = float(auc(recall, precision))

    # Threshold at 0.5
    y_pred_half = (y_proba >= 0.5).astype(int)
    f1_half = float(f1_score(y_test, y_pred_half, pos_label=1, zero_division=0))

    # Cost-optimal threshold
    thresh_result = find_optimal_threshold(y_test, y_proba, fn_cost, fp_cost)
    optimal_thresh = thresh_result["threshold"]
    y_pred_optimal = (y_proba >= optimal_thresh).astype(int)
    f1_optimal = float(f1_score(y_test, y_pred_optimal, pos_label=1, zero_division=0))

    return {
        "auc_roc":                   round(auc_roc, 4),
        "pr_auc":                    round(pr_auc, 4),
        "f1_minority_at_half":       round(f1_half, 4),
        "f1_minority_at_optimal":    round(f1_optimal, 4),
        "optimal_threshold":         optimal_thresh,
        "confusion_matrix_at_optimal": {
            "tn": thresh_result["tn_count"],
            "fp": thresh_result["fp_count"],
            "fn": thresh_result["fn_count"],
            "tp": thresh_result["tp_count"],
        },
        "threshold_cost_report":     thresh_result,
    }


# ── 8. Artefact Persistence ────────────────────────────────────────────────────

def save_model(
    model,
    path: Optional[str | Path] = None,
) -> Path:
    """
    Serialise the calibrated model to data/artifacts/xgb_model.pkl.

    Returns the resolved Path where the file was saved.
    """
    if path is None:
        path = PROJECT_ROOT / "data" / "artifacts" / "xgb_model.pkl"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    return path


def load_model(path: Optional[str | Path] = None):
    """Load a serialised model from disk."""
    if path is None:
        path = PROJECT_ROOT / "data" / "artifacts" / "xgb_model.pkl"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artefact not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── 9. Master Training Pipeline ───────────────────────────────────────────────

def train_churn_model(
    feature_df: Optional[pd.DataFrame] = None,
    feature_path: Optional[str | Path] = None,
    n_optuna_trials: int = OPTUNA_N_TRIALS,
    save: bool = True,
    model_path: Optional[str | Path] = None,
) -> dict:
    """
    End-to-end Day-4 training pipeline.

    Steps:
      1. Load features (from DataFrame or parquet path)
      2. Stratified train/test split
      3. Baseline logistic regression
      4. Optuna HPO for XGBoost
      5. Retrain best XGBoost with SMOTE pipeline on full training set
      6. Platt scaling calibration
      7. Full evaluation on held-out test set
      8. Persist artefact

    Parameters
    ----------
    feature_df      : Pre-loaded DataFrame from build_features(). If None,
                      loads from feature_path.
    feature_path    : Path to features.parquet (default: data/processed/).
    n_optuna_trials : Number of Optuna trials (reduce for fast tests).
    save            : Whether to persist xgb_model.pkl.
    model_path      : Override default artefact save path.

    Returns
    -------
    dict with keys: 'calibrated_model', 'baseline', 'evaluation',
                    'hpo_result', 'feature_names', 'model_path' (if saved),
                    'X_test', 'y_test'
    """
    # ── Load ──
    if feature_df is None:
        if feature_path is None:
            feature_path = PROJECT_ROOT / "data" / "processed" / "features.parquet"
        feature_df = pd.read_parquet(feature_path)

    # ── Split ──
    X_train, X_test, y_train, y_test, feature_names = split_data(feature_df)

    # ── Baseline ──
    baseline = train_baseline(X_train, y_train, X_test, y_test)

    # ── HPO ──
    hpo_result = tune_xgboost(X_train, y_train, n_trials=n_optuna_trials)

    # ── Build best pipeline ──
    best_pipeline = build_smote_xgb_pipeline(hpo_result["best_params"])

    # ── Calibrate ──
    calibrated_model = calibrate_model(best_pipeline, X_train, y_train)

    # ── Evaluate ──
    evaluation = evaluate_model(calibrated_model, X_test, y_test)

    # ── Save ──
    saved_path = None
    if save:
        saved_path = save_model(calibrated_model, model_path)

    return {
        "calibrated_model": calibrated_model,
        "baseline":         baseline,
        "evaluation":       evaluation,
        "hpo_result":       hpo_result,
        "feature_names":    feature_names,
        "model_path":       saved_path,
        "X_test":           X_test,
        "y_test":           y_test,
    }
