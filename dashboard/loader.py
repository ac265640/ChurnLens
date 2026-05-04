"""dashboard/loader.py — cached artifact loading for the Streamlit dashboard."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    model_path  = ROOT / "data" / "artifacts" / "xgb_model.pkl"
    shap_path   = ROOT / "data" / "artifacts" / "shap_values.parquet"
    clv_path    = ROOT / "data" / "processed"  / "customer_clv.parquet"
    feat_path   = ROOT / "data" / "processed"  / "features.parquet"
    prof_path   = ROOT / "data" / "processed"  / "customer_profiles.parquet"
    raw_path    = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    shap_df  = pd.read_parquet(shap_path)
    clv_df   = pd.read_parquet(clv_path)
    feat_df  = pd.read_parquet(feat_path)
    prof_df  = pd.read_parquet(prof_path)

    # Attach customerID from raw CSV (it was dropped during feature engineering)
    if "customerID" not in clv_df.columns and raw_path.exists():
        raw = pd.read_csv(raw_path, usecols=["customerID"])
        # Raw CSV is 7043 rows; all processed parquets preserve that row order
        if len(raw) == len(clv_df):
            clv_df.insert(0, "customerID", raw["customerID"].values)
        if len(raw) == len(feat_df):
            feat_df.insert(0, "customerID", raw["customerID"].values)
        if len(raw) == len(prof_df):
            prof_df.insert(0, "customerID", raw["customerID"].values)
    elif "customerID" not in clv_df.columns:
        # Fallback: generate synthetic IDs
        clv_df.insert(0,  "customerID", [f"CUST-{i:04d}" for i in range(len(clv_df))])
        feat_df.insert(0, "customerID", [f"CUST-{i:04d}" for i in range(len(feat_df))])
        prof_df.insert(0, "customerID", [f"CUST-{i:04d}" for i in range(len(prof_df))])

    return {
        "model":   model,
        "shap_df": shap_df,
        "clv_df":  clv_df,
        "feat_df": feat_df,
        "prof_df": prof_df,
    }

