"""
src/data/loader.py
Schema-validating data loader for the Telco Customer Churn dataset.

Raises descriptive errors on schema violations. Does NOT silently clean data —
cleaning is the responsibility of the preprocessing pipeline (Day 2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd

# ── Schema contract ────────────────────────────────────────────────────────────

EXPECTED_COLUMNS: list[str] = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]

# Dtypes we expect pandas to assign on a clean read.
# TotalCharges is intentionally object — it contains whitespace in raw data.
# We validate it separately via coercibility check, not by asserting float64.
EXPECTED_DTYPES: dict[str, str] = {
    "SeniorCitizen":    "int64",
    "tenure":           "int64",
    "MonthlyCharges":   "float64",
}

CHURN_VALID_VALUES: set[str] = {"Yes", "No"}
ROW_COUNT_RANGE: tuple[int, int] = (7_000, 7_500)


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_raw_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn CSV and validate its schema.

    Parameters
    ----------
    path : str | Path | None
        Path to the CSV file. Defaults to data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
        relative to the project root.

    Returns
    -------
    pd.DataFrame
        The raw, unmodified DataFrame — no cleaning applied.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the resolved path.
    ValueError
        If any schema validation check fails. The message describes exactly
        which check failed and what was found vs. expected.
    """
    # ── 1. Resolve path ────────────────────────────────────────────────────────
    if path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Run `python scripts/download_data.py` to fetch it."
        )

    # ── 2. Load ────────────────────────────────────────────────────────────────
    df = pd.read_csv(path)

    # ── 3. Column presence check ───────────────────────────────────────────────
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_cols   = set(df.columns) - set(EXPECTED_COLUMNS)

    if missing_cols:
        raise ValueError(
            f"Schema violation — missing columns: {sorted(missing_cols)}\n"
            f"Found columns: {sorted(df.columns.tolist())}"
        )
    if extra_cols:
        raise ValueError(
            f"Schema violation — unexpected extra columns: {sorted(extra_cols)}\n"
            "The dataset may be a different version than expected."
        )

    # ── 4. Row count range check ───────────────────────────────────────────────
    n_rows = len(df)
    lo, hi = ROW_COUNT_RANGE
    if not (lo <= n_rows <= hi):
        raise ValueError(
            f"Schema violation — row count {n_rows} is outside expected range "
            f"[{lo}, {hi}]. File may be truncated or a different split."
        )

    # ── 5. Dtype checks for numeric columns ───────────────────────────────────
    for col, expected_dtype in EXPECTED_DTYPES.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"Schema violation — column '{col}' has dtype '{actual_dtype}', "
                f"expected '{expected_dtype}'."
            )

    # ── 6. TotalCharges coercibility check ────────────────────────────────────
    # Raw data has whitespace entries; we assert they are the ONLY non-numeric
    # values (i.e., no garbage strings — just missingness disguised as spaces).
    # Only applies when pandas reads the column as object dtype (as it does with
    # the real dataset). If it's already numeric, it's clean by definition.
    if df["TotalCharges"].dtype == object:
        tc_numeric = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
        non_coercible = df.loc[
            tc_numeric.isna() & (df["TotalCharges"].str.strip() != ""),
            "TotalCharges",
        ]
        if not non_coercible.empty:
            raise ValueError(
                f"Schema violation — 'TotalCharges' contains {len(non_coercible)} "
                f"non-numeric, non-empty value(s): {non_coercible.unique()[:5].tolist()}\n"
                "Expected only whitespace placeholders for missing values."
            )
    else:
        # Column was parsed as numeric — verify no unexpected type
        if str(df["TotalCharges"].dtype) not in ("float64", "int64"):
            raise ValueError(
                f"Schema violation — 'TotalCharges' has unexpected dtype "
                f"'{df['TotalCharges'].dtype}'. Expected object or float64."
            )

    # ── 7. Churn column domain check ──────────────────────────────────────────
    actual_churn_values = set(df["Churn"].dropna().unique())
    unexpected = actual_churn_values - CHURN_VALID_VALUES
    if unexpected:
        raise ValueError(
            f"Schema violation — 'Churn' column contains unexpected values: {unexpected}\n"
            f"Valid values are: {CHURN_VALID_VALUES}"
        )

    return df
