"""
src/features/segmentation.py
Day 3 — Segmentation Layer

Responsibilities:
  1. RFM quartile scoring  (composite 3–12)
  2. K-Means clustering on standardized behavioural features
  3. Optimal-k selection: elbow (inertia) + silhouette score
  4. Bootstrap stability test (Adjusted Rand Index)
  5. Named segment assignment (Champions / Loyal / At-Risk / Hibernating)
  6. Segment profile summary table
  7. Persist customer_profiles.parquet to data/processed/

Nothing in this module touches the Churn column for training — segmentation is
purely unsupervised behavioural discovery.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Features used for K-Means clustering (purely behavioural — no target leakage)
CLUSTER_FEATURES: list[str] = [
    "rfm_recency",
    "rfm_frequency",
    "rfm_monetary",
    "tenure_decay",
    "tenure",
]

# K-Means search range
K_MIN: int = 2
K_MAX: int = 8          # inclusive; we evaluate 2..8

# Bootstrap parameters
BOOTSTRAP_N_RUNS: int = 20
BOOTSTRAP_SEED_START: int = 0

# Segment name labels ordered from best → worst composite RFM rank
SEGMENT_NAMES: list[str] = ["Champions", "Loyal", "At-Risk", "Hibernating"]


# ── 1. RFM Quartile Scoring ────────────────────────────────────────────────────

def score_rfm_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add R_score, F_score, M_score (each 1–4) and rfm_score (3–12).

    Scoring direction (domain semantics):
      R (Recency  = tenure) : ASCENDING  — longer tenure → higher R-score
                              (established customer = lower churn risk = better)
      F (Frequency = services): ASCENDING — more services → higher F-score
      M (Monetary = monthly charges): ASCENDING — higher spend → higher M-score

    Uses pd.qcut with duplicates='drop' to handle ties gracefully.
    """
    df = df.copy()

    def _qscore(series: pd.Series, ascending: bool = True) -> pd.Series:
        """Map series to 1–4 quartile labels, handling ties via 'drop'."""
        labels = [1, 2, 3, 4] if ascending else [4, 3, 2, 1]
        try:
            scored = pd.qcut(series, q=4, labels=labels, duplicates="drop")
        except ValueError:
            # Fewer than 4 unique values — fall back to rank-based scoring
            ranked = series.rank(method="first", ascending=ascending)
            scored = pd.cut(ranked, bins=4, labels=[1, 2, 3, 4])
        return scored.astype(float).fillna(1).astype(int)

    df["R_score"] = _qscore(df["rfm_recency"],  ascending=True)
    df["F_score"] = _qscore(df["rfm_frequency"], ascending=True)
    df["M_score"] = _qscore(df["rfm_monetary"],  ascending=True)
    df["rfm_score"] = df["R_score"] + df["F_score"] + df["M_score"]

    return df


# ── 2. Optimal-k Selection ─────────────────────────────────────────────────────

def select_optimal_k(
    X_scaled: np.ndarray,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    random_state: int = 42,
) -> dict:
    """
    Evaluate K-Means for k in [k_min, k_max].

    Returns a dict with:
      - 'inertias'         : {k: inertia}
      - 'silhouettes'      : {k: silhouette_score}
      - 'optimal_k_sil'    : k with highest silhouette score
      - 'optimal_k_elbow'  : k identified by elbow heuristic (max second-derivative)
      - 'recommended_k'    : final recommendation (silhouette takes priority)
    """
    inertias: dict[int, float] = {}
    silhouettes: dict[int, float] = {}

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias[k] = float(km.inertia_)
        # silhouette_score requires at least 2 labels
        if len(set(labels)) > 1:
            silhouettes[k] = float(silhouette_score(X_scaled, labels, sample_size=min(2000, len(X_scaled))))

    # Elbow heuristic: second derivative of inertia
    ks = sorted(inertias.keys())
    inertia_vals = [inertias[k] for k in ks]
    if len(inertia_vals) >= 3:
        second_deriv = [
            inertia_vals[i - 1] - 2 * inertia_vals[i] + inertia_vals[i + 1]
            for i in range(1, len(inertia_vals) - 1)
        ]
        optimal_k_elbow = ks[1 + int(np.argmax(second_deriv))]
    else:
        optimal_k_elbow = ks[0]

    optimal_k_sil = max(silhouettes, key=silhouettes.get) if silhouettes else k_min

    return {
        "inertias":       inertias,
        "silhouettes":    silhouettes,
        "optimal_k_sil":  optimal_k_sil,
        "optimal_k_elbow": optimal_k_elbow,
        "recommended_k":  optimal_k_sil,   # silhouette is more reliable
    }


# ── 3. K-Means Clustering ──────────────────────────────────────────────────────

def fit_kmeans(
    df: pd.DataFrame,
    k: int,
    feature_cols: Optional[list[str]] = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, StandardScaler, np.ndarray]:
    """
    Fit K-Means on standardized CLUSTER_FEATURES.

    Returns
    -------
    df_out      : input df with 'cluster_id' column added
    km_model    : fitted KMeans object
    scaler      : fitted StandardScaler (needed for transform at inference)
    X_scaled    : the scaled feature matrix used for fitting
    """
    if feature_cols is None:
        feature_cols = CLUSTER_FEATURES

    # Guard: only keep features that actually exist in df
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        raise ValueError(
            f"None of the requested cluster features found in df. "
            f"Requested: {feature_cols}. Available: {df.columns.tolist()}"
        )

    X = df[available].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    df = df.copy()
    df["cluster_id"] = km.fit_predict(X_scaled)

    return df, km, scaler, X_scaled


# ── 4. Bootstrap Stability Test ────────────────────────────────────────────────

def bootstrap_stability(
    X_scaled: np.ndarray,
    k: int,
    n_runs: int = BOOTSTRAP_N_RUNS,
    seed_start: int = BOOTSTRAP_SEED_START,
) -> dict:
    """
    Run K-Means n_runs times with different random seeds.
    Compute pairwise Adjusted Rand Index (ARI) between all run-pairs.

    Returns
    -------
    dict with 'ari_mean', 'ari_std', 'ari_min', 'stable' (bool: mean ARI ≥ 0.7)
    """
    all_labels: list[np.ndarray] = []
    for seed in range(seed_start, seed_start + n_runs):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        all_labels.append(km.fit_predict(X_scaled))

    ari_scores: list[float] = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            ari_scores.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    ari_arr = np.array(ari_scores)
    mean_ari = float(np.mean(ari_arr))
    std_ari  = float(np.std(ari_arr))
    min_ari  = float(np.min(ari_arr))

    return {
        "ari_mean": round(mean_ari, 4),
        "ari_std":  round(std_ari,  4),
        "ari_min":  round(min_ari,  4),
        "n_runs":   n_runs,
        "k":        k,
        "stable":   mean_ari >= 0.70,
    }


# ── 5. Named Segment Assignment ────────────────────────────────────────────────

def _label_by_rfm_score(score: int) -> str:
    """
    Map a composite rfm_score (3–12) to one of the 4 canonical segment names.

    Thresholds chosen so the full 3–12 integer range is always covered:
      12–10  → Champions    (top quartile behaviour)
       9–7   → Loyal Active (above-median engagement)
       6–5   → At-Risk      (below-median, some engagement)
       4–3   → Hibernating  (lowest engagement)
    """
    if score >= 10:
        return "Champions"
    elif score >= 7:
        return "Loyal Active"
    elif score >= 5:
        return "At-Risk"
    else:
        return "Hibernating"


def assign_segment_names(
    df: pd.DataFrame,
    km_model: KMeans,
    feature_cols: Optional[list[str]] = None,
    scaler: Optional[StandardScaler] = None,
) -> pd.DataFrame:
    """
    Assign human-readable segment names to every customer.

    Priority logic:
    1. If 'rfm_score' column is present (added by score_rfm_quartiles),
       apply threshold-based labelling via _label_by_rfm_score().
       This guarantees all 4 segment names appear regardless of k.
    2. Fallback: rank cluster centroids by composite RFM and map to
       SEGMENT_NAMES in order (original cluster-rank approach). Limited
       to producing at most k distinct names.

    The rfm_score path is preferred because it is fully deterministic,
    produces all 4 canonical labels on any realistic dataset, and is
    independent of the K-Means random seed.
    """
    df = df.copy()

    # ── Path 1: rfm_score threshold labelling (preferred) ─────────────────────
    if "rfm_score" in df.columns:
        df["segment"] = df["rfm_score"].apply(_label_by_rfm_score)
        return df

    # ── Path 2: cluster-rank fallback ─────────────────────────────────────────
    if feature_cols is None:
        feature_cols = CLUSTER_FEATURES

    available = [c for c in feature_cols if c in df.columns]

    centroids_scaled = km_model.cluster_centers_
    if scaler is not None:
        centroids_orig = scaler.inverse_transform(centroids_scaled)
        centroid_df = pd.DataFrame(centroids_orig, columns=available)
    else:
        centroid_df = pd.DataFrame(centroids_scaled, columns=available)

    rfm_proxy_cols = [
        c for c in ("rfm_recency", "rfm_frequency", "rfm_monetary")
        if c in centroid_df.columns
    ]
    centroid_df["composite"] = (
        centroid_df[rfm_proxy_cols].mean(axis=1) if rfm_proxy_cols
        else centroid_df[available[0]]
    )

    centroid_df["rank"] = (
        centroid_df["composite"].rank(ascending=False, method="first").astype(int) - 1
    )

    n_clusters = km_model.n_clusters
    names = SEGMENT_NAMES[:n_clusters]
    while len(names) < n_clusters:
        names.append(f"Segment_{len(names) + 1}")

    cluster_to_name = {
        cluster_id: names[rank]
        for cluster_id, rank in centroid_df["rank"].items()
    }

    df["segment"] = df["cluster_id"].map(cluster_to_name)
    return df


# ── 6. Segment Profile Summary ─────────────────────────────────────────────────

def build_segment_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-segment summary statistics.

    Returns a DataFrame indexed by segment name with columns:
      n_customers, pct_of_total, mean_tenure, mean_monthly_charges,
      mean_rfm_score (if rfm_score present), mean_services (rfm_frequency),
      churn_rate_pct (if Churn present).

    Gracefully handles DataFrames that have not been passed through
    score_rfm_quartiles() — all agg columns are checked for existence first.
    """
    # Use first non-segment, non-cluster column as count proxy
    count_col = next(
        (c for c in df.columns if c not in ("segment", "cluster_id")),
        df.columns[0],
    )

    # Build agg spec dynamically — only include columns that exist
    agg_spec: dict[str, tuple] = {
        "n_customers": (count_col, "count"),
    }
    if "tenure" in df.columns:
        agg_spec["mean_tenure"] = ("tenure", "mean")
    if "MonthlyCharges" in df.columns:
        agg_spec["mean_monthly"] = ("MonthlyCharges", "mean")
    if "rfm_score" in df.columns:
        agg_spec["mean_rfm_score"] = ("rfm_score", "mean")
    if "rfm_frequency" in df.columns:
        agg_spec["mean_services"] = ("rfm_frequency", "mean")

    profile = df.groupby("segment").agg(**agg_spec).round(2)

    profile["pct_of_total"] = (
        profile["n_customers"] / profile["n_customers"].sum() * 100
    ).round(1)

    if "Churn" in df.columns:
        churn_rate = df.groupby("segment")["Churn"].mean().round(4) * 100
        profile["churn_rate_pct"] = churn_rate.round(1)

    sort_col = "mean_rfm_score" if "mean_rfm_score" in profile.columns else "n_customers"
    profile = profile.sort_values(sort_col, ascending=False)
    return profile


# ── 7. Master Pipeline ─────────────────────────────────────────────────────────

def run_segmentation(
    feature_df: Optional[pd.DataFrame] = None,
    feature_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    k: Optional[int] = None,
    save: bool = True,
) -> dict:
    """
    Full Day-3 segmentation pipeline. Returns a results dict.

    Parameters
    ----------
    feature_df   : Pre-loaded feature DataFrame (from build_features()).
                   If None, loads from feature_path.
    feature_path : Path to features.parquet. Defaults to data/processed/features.parquet.
    output_path  : Where to save customer_profiles.parquet.
    k            : Force a specific k. If None, auto-selects via silhouette.
    save         : Whether to persist customer_profiles.parquet.

    Returns
    -------
    dict with keys: 'profiles_df', 'segment_summary', 'k_selection',
                    'stability', 'km_model', 'scaler'
    """
    # ── Load features ──
    if feature_df is None:
        if feature_path is None:
            feature_path = PROJECT_ROOT / "data" / "processed" / "features.parquet"
        feature_df = pd.read_parquet(feature_path)

    df = feature_df.copy()

    # ── RFM quartile scoring ──
    df = score_rfm_quartiles(df)

    # ── Build scaled feature matrix for clustering ──
    available_cluster_cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    X = df[available_cluster_cols].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Select optimal k ──
    k_selection = select_optimal_k(X_scaled)
    if k is None:
        k = k_selection["recommended_k"]

    # ── Fit K-Means ──
    df, km_model, _, _ = fit_kmeans(df, k=k, feature_cols=available_cluster_cols)

    # ── Bootstrap stability ──
    stability = bootstrap_stability(X_scaled, k=k)

    # ── Assign segment names ──
    df = assign_segment_names(df, km_model, feature_cols=available_cluster_cols, scaler=scaler)

    # ── Segment profile ──
    segment_summary = build_segment_profile(df)

    # ── Persist ──
    if save:
        if output_path is None:
            output_path = PROJECT_ROOT / "data" / "processed" / "customer_profiles.parquet"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

    return {
        "profiles_df":      df,
        "segment_summary":  segment_summary,
        "k_selection":      k_selection,
        "stability":        stability,
        "km_model":         km_model,
        "scaler":           scaler,
        "k_used":           k,
    }
