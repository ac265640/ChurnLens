"""
src/models/clv.py
Day 5 — CLV Estimation + Priority Matrix

Components:
  1. DCF CLV — PV-annuity formula (monthly_revenue, expected tenure, discount rate)
  2. BG/NBD CLV via `lifetimes` library (Gamma-Gamma for monetary value)
  3. 2×2 Priority Matrix: Save / Nurture / Accept / Monitor
  4. Revenue at Risk: churn_probability × CLV per customer
  5. CLV × Churn Risk heatmap data (n×n quantile-binned grid)
  6. Master pipeline: build_clv_table() → data/processed/customer_clv.parquet

NOTE on BG/NBD: This model is designed for non-contractual repeat-purchase
settings. Telco is contractual; we apply BG/NBD here per the project spec
and treat its output as a relative CLV index, not literal transaction counts.
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

# ── Optional lifetimes import ──────────────────────────────────────────────────
try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    _LIFETIMES_AVAILABLE = True
except ImportError:
    _LIFETIMES_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MONTHLY_DISCOUNT_RATE: float = 0.01   # 1 % / month ≈ 12.7 % annual
TIME_HORIZON_MONTHS:   int   = 12     # BG/NBD prediction window
CLV_HIGH_PERCENTILE:   int   = 50     # median split for priority matrix
CHURN_HIGH_THRESHOLD:  float = 0.50   # prob ≥ this → High Risk
HEATMAP_BINS:          int   = 3      # Low / Medium / High bins
HEATMAP_LABELS: list[str]    = ["Low", "Medium", "High"]

# Priority quadrant labels
SAVE    = "Save"     # High CLV + High Risk  → intervene immediately
NURTURE = "Nurture"  # High CLV + Low Risk   → grow the relationship
ACCEPT  = "Accept"   # Low CLV  + High Risk  → accept probable loss
MONITOR = "Monitor"  # Low CLV  + Low Risk   → routine watch


# ── 1. Expected Tenure from Churn Probability ─────────────────────────────────

def expected_tenure_from_churn_prob(
    churn_prob: float | np.ndarray,
    min_months: float = 1.0,
    max_months: float = 120.0,
) -> float | np.ndarray:
    """
    Convert an observation-window churn probability to expected remaining
    tenure (months) via a geometric-distribution assumption.

    Under the geometric model:
        monthly_churn_rate  p_m = 1 − (1 − churn_prob)^(1/12)
        expected_tenure         = 1 / p_m

    The (1/12) exponent assumes the model's observation window ≈ 12 months.

    Parameters
    ----------
    churn_prob : P(churn) in the observation window, scalar or array.
    min_months : Floor (default 1 month).
    max_months : Cap  (default 120 months / 10 years).

    Returns
    -------
    Expected remaining tenure in months — same shape as input.
    """
    p = np.clip(np.asarray(churn_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    monthly_p = 1.0 - (1.0 - p) ** (1.0 / 12.0)
    tenure = np.clip(1.0 / monthly_p, min_months, max_months)
    return float(tenure) if tenure.ndim == 0 else tenure


# ── 2. DCF CLV ────────────────────────────────────────────────────────────────

def compute_dcf_clv(
    monthly_revenue: float | np.ndarray,
    expected_tenure_months: float | np.ndarray,
    monthly_discount_rate: float = MONTHLY_DISCOUNT_RATE,
) -> float | np.ndarray:
    """
    Discounted Cash-Flow CLV using the present-value annuity formula:

        CLV = monthly_revenue × [(1 − (1 + r)^−T) / r]

    where T = expected_tenure_months (floored at 1), r = monthly discount rate.
    If r == 0, returns monthly_revenue × T (no discounting).

    Parameters
    ----------
    monthly_revenue        : Monthly charge per customer (scalar or array).
    expected_tenure_months : Expected remaining lifetime in months.
    monthly_discount_rate  : Monthly discount rate r (default 0.01).

    Returns
    -------
    CLV — same shape as inputs (broadcast-compatible).
    """
    T   = np.maximum(np.asarray(expected_tenure_months, dtype=float), 1.0)
    rev = np.asarray(monthly_revenue, dtype=float)

    if monthly_discount_rate == 0.0:
        result = rev * T
    else:
        r         = float(monthly_discount_rate)
        pv_factor = (1.0 - (1.0 + r) ** (-T)) / r
        result    = rev * pv_factor

    return float(result) if result.ndim == 0 else result


# ── 3. BG/NBD Model ───────────────────────────────────────────────────────────

def fit_bgnbd_model(
    frequency:  pd.Series | np.ndarray,
    recency:    pd.Series | np.ndarray,
    T:          pd.Series | np.ndarray,
    penalizer:  float = 0.01,
) -> "BetaGeoFitter":
    """
    Fit a BG/NBD model on RFM inputs using the `lifetimes` library.

    Parameters
    ----------
    frequency : Repeat-event count per customer (≥ 0 integers).
    recency   : Months from first to last event (0 ≤ recency ≤ T).
    T         : Total observation window per customer in months.
    penalizer : L2 regularisation coefficient (default 0.01).

    Returns
    -------
    Fitted BetaGeoFitter instance.
    """
    if not _LIFETIMES_AVAILABLE:
        raise RuntimeError("lifetimes not installed. Run: pip install lifetimes")

    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(
        np.asarray(frequency, dtype=float),
        np.asarray(recency,   dtype=float),
        np.asarray(T,         dtype=float),
    )
    return bgf


def compute_bgnbd_clv(
    frequency:            pd.Series | np.ndarray,
    recency:              pd.Series | np.ndarray,
    T:                    pd.Series | np.ndarray,
    monetary_value:       pd.Series | np.ndarray,
    bgf:                  "BetaGeoFitter",
    time_horizon_months:  int   = TIME_HORIZON_MONTHS,
    monthly_discount_rate: float = MONTHLY_DISCOUNT_RATE,
) -> np.ndarray:
    """
    BG/NBD + Gamma-Gamma CLV for a future `time_horizon_months` window.

    Steps:
      1. BG/NBD → expected future transactions over the horizon.
      2. Gamma-Gamma → expected monetary value per transaction
         (fitted on customers with ≥ 1 repeat purchase; others use raw monetary).
      3. CLV = expected_transactions × avg_monetary × PV_annuity_factor.

    Returns
    -------
    np.ndarray of non-negative CLV values (one per customer).
    """
    if not _LIFETIMES_AVAILABLE:
        raise RuntimeError("lifetimes not installed. Run: pip install lifetimes")

    freq = np.asarray(frequency,      dtype=float)
    rec  = np.asarray(recency,        dtype=float)
    obs  = np.asarray(T,              dtype=float)
    mon  = np.asarray(monetary_value, dtype=float)

    # Expected transactions over the future window
    exp_tx = bgf.predict(
        t=time_horizon_months,
        frequency=freq,
        recency=rec,
        T=obs,
    )

    # Gamma-Gamma monetary model (requires customers with ≥ 1 repeat)
    mask = freq > 0
    if mask.sum() >= 10:
        ggf = GammaGammaFitter(penalizer_coef=0.01)
        ggf.fit(freq[mask], mon[mask])
        exp_mon = ggf.conditional_expected_average_profit(freq, mon)
    else:
        exp_mon = mon  # fallback: raw monetary value

    # Discount over the horizon
    r  = float(monthly_discount_rate)
    pv = (1.0 - (1.0 + r) ** (-time_horizon_months)) / r if r > 0 else float(time_horizon_months)

    clv = exp_tx * exp_mon * pv / time_horizon_months
    return np.maximum(np.asarray(clv, dtype=float), 0.0)


# ── 4. Priority Matrix ────────────────────────────────────────────────────────

def assign_priority_quadrant(
    clv:              pd.Series | np.ndarray,
    churn_prob:       pd.Series | np.ndarray,
    clv_threshold:    Optional[float] = None,
    churn_threshold:  float = CHURN_HIGH_THRESHOLD,
) -> np.ndarray:
    """
    Assign each customer to a retention-priority quadrant.

    2×2 Matrix:
    ┌────────────────────────────────────────────────┐
    │                │ Churn Risk Low │ Churn Risk High │
    │ CLV High  ──▶  │   NURTURE      │   SAVE          │
    │ CLV Low   ──▶  │   MONITOR      │   ACCEPT        │
    └────────────────────────────────────────────────┘

    Parameters
    ----------
    clv             : CLV per customer.
    churn_prob      : Churn probability in [0, 1].
    clv_threshold   : Split point; defaults to median CLV.
    churn_threshold : Probability ≥ this → High Risk (default 0.5).

    Returns
    -------
    np.ndarray of strings: 'Save' | 'Nurture' | 'Accept' | 'Monitor'.
    """
    clv_arr   = np.asarray(clv,        dtype=float)
    churn_arr = np.asarray(churn_prob, dtype=float)

    if clv_threshold is None:
        clv_threshold = float(np.median(clv_arr))

    high_clv  = clv_arr   >= clv_threshold
    high_risk = churn_arr >= churn_threshold

    return np.where(
        high_clv  &  high_risk,  SAVE,
        np.where(
            high_clv  & ~high_risk, NURTURE,
            np.where(
                ~high_clv &  high_risk, ACCEPT,
                MONITOR
            )
        )
    )


# ── 5. Revenue at Risk ────────────────────────────────────────────────────────

def compute_revenue_at_risk(
    clv:        pd.Series | np.ndarray,
    churn_prob: pd.Series | np.ndarray,
) -> np.ndarray:
    """
    Expected revenue lost if the customer churns:

        revenue_at_risk = CLV × churn_probability

    This is the single most actionable number for retention budget sizing:
    intervene whenever revenue_at_risk > cost_of_retention_action.

    Returns
    -------
    np.ndarray of non-negative floats.
    """
    clv_arr   = np.asarray(clv,        dtype=float)
    churn_arr = np.clip(np.asarray(churn_prob, dtype=float), 0.0, 1.0)
    return np.maximum(clv_arr * churn_arr, 0.0)


# ── 6. CLV × Churn Heatmap Data ──────────────────────────────────────────────

def build_clv_heatmap_data(
    clv:        pd.Series | np.ndarray,
    churn_prob: pd.Series | np.ndarray,
    n_bins:     int       = HEATMAP_BINS,
    labels:     list[str] = HEATMAP_LABELS,
) -> pd.DataFrame:
    """
    Build an (n_bins × n_bins) pivot table for the CLV × Churn Risk heatmap.

    Both CLV and churn_prob are binned into `n_bins` quantile-based groups.
    Each cell contains the mean revenue_at_risk for customers in that bin.

    Parameters
    ----------
    clv        : CLV per customer.
    churn_prob : Churn probability in [0, 1].
    n_bins     : Number of quantile bins (default 3 → Low / Medium / High).
    labels     : Bin label strings (length must equal n_bins).

    Returns
    -------
    pd.DataFrame  shape (n_bins, n_bins):
      index   → CLV bins   (Low → High)
      columns → Churn bins (Low → High)
      values  → mean revenue_at_risk
    """
    clv_arr   = np.asarray(clv,        dtype=float)
    churn_arr = np.asarray(churn_prob, dtype=float)
    rar       = compute_revenue_at_risk(clv_arr, churn_arr)

    clv_bin   = pd.qcut(clv_arr,   q=n_bins, labels=labels[:n_bins], duplicates="drop")
    churn_bin = pd.qcut(churn_arr, q=n_bins, labels=labels[:n_bins], duplicates="drop")

    return (
        pd.DataFrame({"clv_bin": clv_bin, "churn_bin": churn_bin, "rar": rar})
        .pivot_table(index="clv_bin", columns="churn_bin",
                     values="rar", aggfunc="mean", observed=True)
        .fillna(0.0)
    )


# ── 7. Master CLV Pipeline ────────────────────────────────────────────────────

def build_clv_table(
    profiles_path: Optional[str | Path] = None,
    features_path: Optional[str | Path] = None,
    model_path:    Optional[str | Path] = None,
    output_path:   Optional[str | Path] = None,
    use_bgnbd:     bool = True,
) -> pd.DataFrame:
    """
    End-to-end Day-5 CLV pipeline.

    Steps:
      1. Load customer_profiles.parquet + features.parquet.
      2. Load calibrated XGBoost model → predict churn probabilities.
      3. Compute DCF CLV per customer.
      4. Optionally compute BG/NBD CLV (blended with DCF for robustness).
      5. Assign priority quadrant (Save / Nurture / Accept / Monitor).
      6. Compute revenue_at_risk per customer.
      7. Attach CLV × churn heatmap bin labels.
      8. Persist enriched table to data/processed/customer_clv.parquet.

    Parameters
    ----------
    profiles_path : Path to customer_profiles.parquet.
    features_path : Path to features.parquet (used for model input).
    model_path    : Path to xgb_model.pkl.
    output_path   : Output parquet path (default: data/processed/customer_clv.parquet).
    use_bgnbd     : Whether to fit and include BG/NBD CLV column.

    Returns
    -------
    pd.DataFrame — one row per customer with CLV + priority enrichment.
    """
    # ── Resolve paths ──
    if profiles_path is None:
        profiles_path = PROJECT_ROOT / "data" / "processed" / "customer_profiles.parquet"
    if features_path is None:
        features_path = PROJECT_ROOT / "data" / "processed" / "features.parquet"
    if model_path is None:
        model_path = PROJECT_ROOT / "data" / "artifacts" / "xgb_model.pkl"
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "processed" / "customer_clv.parquet"

    profiles_path = Path(profiles_path)
    features_path = Path(features_path)
    model_path    = Path(model_path)
    output_path   = Path(output_path)

    # ── Load data ──
    profiles = pd.read_parquet(profiles_path)
    features = pd.read_parquet(features_path)

    # ── Predict churn probabilities ──
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    from src.models.churn_model import NON_FEATURE_COLS
    feature_cols = [
        c for c in features.columns
        if c not in NON_FEATURE_COLS and features[c].dtype != object
    ]
    X = features[feature_cols].values.astype(float)
    churn_prob = model.predict_proba(X)[:, 1]

    # ── DCF CLV ──
    monthly_revenue    = features["MonthlyCharges"].values
    expected_tenure    = expected_tenure_from_churn_prob(churn_prob)
    clv_dcf            = compute_dcf_clv(monthly_revenue, expected_tenure)

    # ── BG/NBD CLV (optional) ──
    clv_bgnbd = np.zeros(len(features))
    if use_bgnbd and _LIFETIMES_AVAILABLE:
        try:
            freq = np.round(features["rfm_frequency"].values).astype(float)
            rec  = features["rfm_recency"].values.astype(float)
            obs  = np.full(len(features), features["tenure"].max(), dtype=float)
            # BG/NBD constraint: recency must be ≤ T and 0 when frequency == 0
            rec  = np.minimum(rec, obs)
            rec  = np.where(freq == 0, 0.0, rec)
            bgf  = fit_bgnbd_model(freq, rec, obs)
            clv_bgnbd = compute_bgnbd_clv(
                freq, rec, obs,
                monetary_value=monthly_revenue,
                bgf=bgf,
                time_horizon_months=TIME_HORIZON_MONTHS,
            )
        except Exception as e:
            warnings.warn(f"BG/NBD fitting failed ({e}); clv_bgnbd set to 0.", stacklevel=2)

    # ── Priority quadrant ──
    quadrant = assign_priority_quadrant(clv_dcf, churn_prob)

    # ── Revenue at risk ──
    rar = compute_revenue_at_risk(clv_dcf, churn_prob)

    # ── Heatmap bins ──
    clv_bin   = pd.qcut(clv_dcf,   q=HEATMAP_BINS,
                        labels=HEATMAP_LABELS, duplicates="drop")
    churn_bin = pd.qcut(churn_prob, q=HEATMAP_BINS,
                        labels=HEATMAP_LABELS, duplicates="drop")

    # ── Assemble output table ──
    out = profiles.copy()
    out["churn_probability"]     = np.round(churn_prob, 4)
    out["expected_tenure_months"] = np.round(expected_tenure, 2)
    out["clv_dcf"]               = np.round(clv_dcf, 2)
    out["clv_bgnbd"]             = np.round(clv_bgnbd, 2)
    out["revenue_at_risk"]       = np.round(rar, 2)
    out["priority_quadrant"]     = quadrant
    out["clv_bin"]               = clv_bin.astype(str)
    out["churn_risk_bin"]        = churn_bin.astype(str)

    # ── Persist ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    return out
