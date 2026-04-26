"""
src/data/preprocessor.py
EDA audit + leak-free feature engineering pipeline.

Produces a model-ready feature matrix from the raw Telco CSV.
Cleaning decisions are explicit and documented inline.

Day 2 scope:
  - Missing value audit
  - Class imbalance quantification
  - Categorical encoding (ordinal / one-hot)
  - RFM feature construction
  - Temporal decay feature
  - Feature correlation audit helpers
  - Saves processed features to data/processed/features.parquet
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.loader import load_raw_data

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Temporal decay rate (λ). Chosen so that exp(-λ×72) ≈ 0.07:
# a 6-year customer retains only 7% of "newness risk".
DECAY_LAMBDA: float = 0.04

# Columns dropped before feature engineering (ID / zero-info / post-hoc)
COLS_TO_DROP: list[str] = ["customerID"]

# Binary yes/no columns mapped to 1/0
BINARY_YES_NO_COLS: list[str] = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

# Columns with yes/no/no-phone-service or yes/no/no-internet-service → map cleanly
TERNARY_SERVICE_COLS: list[str] = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Ordinal columns — order is meaningful; encoded as integers
ORDINAL_MAPPINGS: dict[str, dict[str, int]] = {
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
}

# Nominal columns — one-hot encoded (low cardinality, no order implied)
NOMINAL_COLS: list[str] = [
    "gender",
    "InternetService",
    "PaymentMethod",
]

# Service columns used for RFM "Frequency" count
SERVICE_COLS_FOR_FREQUENCY: list[str] = [
    "PhoneService",      # already binary at this stage
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


# ── EDA helpers ────────────────────────────────────────────────────────────────

def audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarising missing value counts and percentages.

    Includes TotalCharges whitespace entries (treated as missing in the raw data).
    """
    report: dict[str, dict] = {}

    for col in df.columns:
        null_count = int(df[col].isna().sum())

        # TotalCharges has whitespace-disguised NaNs — count those too
        whitespace_count = 0
        if df[col].dtype == object:
            whitespace_count = int((df[col].str.strip() == "").sum())

        total_missing = null_count + whitespace_count
        report[col] = {
            "null_count":       null_count,
            "whitespace_count": whitespace_count,
            "total_missing":    total_missing,
            "pct_missing":      round(100 * total_missing / len(df), 3),
            "dtype":            str(df[col].dtype),
        }

    return pd.DataFrame(report).T.sort_values("total_missing", ascending=False)


def quantify_class_imbalance(df: pd.DataFrame, target_col: str = "Churn") -> dict:
    """
    Return a dict with class counts, ratios, and an imbalance severity label.

    Severity thresholds (industry convention):
      - <10 % minority  → severe
      - 10–30 %         → moderate
      - >30 %           → mild
    """
    counts   = df[target_col].value_counts()
    total    = len(df)
    minority_pct = 100 * counts.min() / total

    if minority_pct < 10:
        severity = "severe"
    elif minority_pct < 30:
        severity = "moderate"
    else:
        severity = "mild"

    return {
        "class_counts":   counts.to_dict(),
        "total":          total,
        "minority_pct":   round(minority_pct, 2),
        "majority_pct":   round(100 - minority_pct, 2),
        "imbalance_ratio": round(counts.max() / counts.min(), 2),
        "severity":       severity,
    }


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute TotalCharges whitespace entries with MonthlyCharges × tenure.

    Rationale: whitespace entries correspond to new customers (tenure=0)
    with no accumulated charges. MonthlyCharges × 0 = 0 is the correct value,
    but we use MonthlyCharges × max(tenure, 1) to avoid a hard zero.

    Note: we operate on a copy to preserve the raw DataFrame.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")

    missing_mask = df["TotalCharges"].isna()
    df.loc[missing_mask, "TotalCharges"] = (
        df.loc[missing_mask, "MonthlyCharges"]
        * df.loc[missing_mask, "tenure"].clip(lower=1)
    )
    return df


# ── Encoding ─────────────────────────────────────────────────────────────────

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map Churn Yes→1, No→0. Returns df with 'Churn' as int8."""
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("int8")
    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Map gender Male→1, Female→0."""
    df = df.copy()
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).astype("int8")
    return df


def encode_binary_yes_no(df: pd.DataFrame) -> pd.DataFrame:
    """Map Yes→1, No→0 for all BINARY_YES_NO_COLS."""
    df = df.copy()
    for col in BINARY_YES_NO_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0}).astype("int8")
    return df


def encode_ternary_service_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map ternary service columns:
      Yes → 1, No → 0, No phone/internet service → 0
    Semantics: 0 means "does not have this service" regardless of reason.
    """
    df = df.copy()
    mapping = {"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0}
    for col in TERNARY_SERVICE_COLS:
        df[col] = df[col].map(mapping).astype("int8")
    return df


def encode_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ORDINAL_MAPPINGS to preserve ordinal relationships."""
    df = df.copy()
    for col, mapping in ORDINAL_MAPPINGS.items():
        df[col] = df[col].map(mapping).astype("int8")
    return df


def encode_nominal(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode NOMINAL_COLS. Drop first dummy to avoid multicollinearity.
    Column names: <original_col>_<value>, spaces replaced with underscores.
    """
    df = pd.get_dummies(
        df,
        columns=NOMINAL_COLS,
        drop_first=True,
        dtype="int8",
    )
    # Normalise column names: replace spaces and hyphens
    df.columns = [c.replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


# ── RFM Feature Engineering ───────────────────────────────────────────────────

def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct three RFM proxy features from existing columns.

    Recency  → tenure (months active; higher = less recently acquired)
    Frequency → count of active services (already binary-encoded at call time)
    Monetary  → MonthlyCharges (current revenue per customer)

    These features are added alongside the originals (no dropping here).
    """
    df = df.copy()

    # Recency: raw tenure is already meaningful — we rename for explicitness
    df["rfm_recency"] = df["tenure"].astype("float32")

    # Frequency: sum of all binary service indicators
    # Guard: use only columns that are still present after encoding
    available_service_cols = [c for c in SERVICE_COLS_FOR_FREQUENCY if c in df.columns]
    df["rfm_frequency"] = df[available_service_cols].sum(axis=1).astype("float32")

    # Monetary: monthly charges directly
    df["rfm_monetary"] = df["MonthlyCharges"].astype("float32")

    return df


# ── Temporal Decay Feature ────────────────────────────────────────────────────

def build_decay_feature(df: pd.DataFrame, lam: float = DECAY_LAMBDA) -> pd.DataFrame:
    """
    Add exponential decay feature: tenure_decay = exp(-λ × tenure).

    Interpretation: value close to 1.0 → very new customer (high risk signal).
    Value close to 0.0 → long-tenured customer.
    λ = 0.04 → half-life ≈ 17 months.
    """
    df = df.copy()
    df["tenure_decay"] = np.exp(-lam * df["tenure"]).astype("float32")
    return df


# ── Correlation Audit ─────────────────────────────────────────────────────────

def correlation_audit(
    df: pd.DataFrame,
    target_col: str = "Churn",
    threshold: float = 0.85,
) -> dict:
    """
    Return a dict with:
      - 'high_pairs': feature pairs with |correlation| >= threshold (potential redundancy)
      - 'target_corr': Pearson correlation of all numeric features with target
    """
    numeric_df = df.select_dtypes(include=[np.number])

    corr_matrix = numeric_df.corr(method="pearson")

    # Extract upper-triangle pairs above threshold (excluding self-correlation)
    high_pairs = []
    cols = corr_matrix.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            val = corr_matrix.loc[c1, c2]
            if abs(val) >= threshold and c1 != target_col and c2 != target_col:
                high_pairs.append((c1, c2, round(float(val), 4)))

    target_corr = (
        corr_matrix[target_col]
        .drop(target_col, errors="ignore")
        .sort_values(key=abs, ascending=False)
        .round(4)
        .to_dict()
    )

    return {
        "high_pairs":   high_pairs,
        "target_corr":  target_corr,
        "threshold":    threshold,
    }


# ── Master Pipeline ───────────────────────────────────────────────────────────

def build_features(
    raw_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    save: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Full Day-2 feature engineering pipeline. Returns (feature_df, eda_report).

    Parameters
    ----------
    raw_path    : Path to raw CSV. Defaults to data/raw/ standard location.
    output_path : Where to save features.parquet. Defaults to data/processed/.
    save        : Whether to persist the parquet file.

    Returns
    -------
    (df, eda_report)
        df         — engineered feature matrix including 'Churn' target column
        eda_report — dict with missing audit, imbalance stats, correlation audit
    """
    # Step 1: Load raw validated data
    df = load_raw_data(raw_path)

    # Step 2: EDA audit (pre-cleaning — on raw df)
    missing_report    = audit_missing(df)
    imbalance_report  = quantify_class_imbalance(df, target_col="Churn")

    # Step 3: Clean
    df = clean_total_charges(df)

    # Step 4: Drop zero-info columns
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")

    # Step 5: Encode target
    df = encode_target(df)

    # Step 6: Encode features
    df = encode_gender(df)
    df = encode_binary_yes_no(df)
    df = encode_ternary_service_cols(df)
    df = encode_ordinal(df)

    # Step 7: RFM + decay (before one-hot so SERVICE_COLS_FOR_FREQUENCY are intact)
    df = build_rfm_features(df)
    df = build_decay_feature(df)

    # Step 8: One-hot encode remaining nominals
    df = encode_nominal(df)

    # Step 9: Correlation audit (post-encoding)
    corr_report = correlation_audit(df, target_col="Churn")

    # Step 10: Persist
    if save:
        if output_path is None:
            output_path = PROJECT_ROOT / "data" / "processed" / "features.parquet"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

    eda_report = {
        "missing":     missing_report,
        "imbalance":   imbalance_report,
        "correlation": corr_report,
    }

    return df, eda_report
