"""
dashboard/app.py
Day 7 — ChurnLens Streamlit Dashboard

6 Tabs:
  1. Overview          — KPIs, churn distribution, priority matrix
  2. RFM Segments      — Segment scatter, CLV by segment
  3. Cluster Profiles  — Cluster breakdown table + bar chart
  4. Churn Model       — Live threshold slider, confusion matrix, PR curve, calibration
  5. CLV Analysis      — CLV×churn heatmap, revenue-at-risk distribution
  6. Action Center     — Customer lookup, churn gauge, SHAP waterfall, recommendation

Run with:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of where Streamlit is launched from
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from dashboard.loader import load_artifacts
from dashboard.charts import (
    churn_distribution_fig,
    churn_gauge_fig,
    clv_heatmap_fig,
    calibration_fig,
    confusion_matrix_fig,
    dependence_fig,
    pr_curve_fig,
    priority_matrix_fig,
    rfm_segment_fig,
    shap_beeswarm_fig,
    shap_waterfall_fig,
    PALETTE,
)
from src.models.shap_explainer import (
    compute_calibration,
    compute_confusion_matrix,
    compute_pr_curve,
    global_shap_summary,
    shap_dependence_data,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnLens — Retention Intelligence",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background-color: #0F1117; color: #E0E0E0; }
  .block-container { padding: 1.5rem 2rem; }

  /* KPI cards */
  .kpi-card {
    background: linear-gradient(135deg, #1E2130 0%, #252840 100%);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #2d3050;
    text-align: center;
  }
  .kpi-value { font-size: 2.2rem; font-weight: 700; margin: 0; }
  .kpi-label { font-size: 0.85rem; color: #8890b5; margin-top: 4px; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #1E2130; border-radius: 10px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 0.5rem 1.2rem; color: #8890b5; font-weight: 500; }
  .stTabs [aria-selected="true"] { background: #6C63FF !important; color: white !important; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #1E2130; border-right: 1px solid #2d3050; }
  section[data-testid="stSidebar"] .block-container { padding: 1rem; }

  /* Metric override */
  [data-testid="stMetric"] { background: #1E2130; border-radius: 10px; padding: 1rem; border: 1px solid #2d3050; }
  [data-testid="stMetricLabel"] { color: #8890b5 !important; }

  /* Section headers */
  .section-header {
    font-size: 1.1rem; font-weight: 600;
    color: #6C63FF; border-left: 3px solid #6C63FF;
    padding-left: 10px; margin: 1rem 0 0.5rem 0;
  }

  /* Action badge */
  .badge-save    { background: #FF6B6B22; color: #FF6B6B; border: 1px solid #FF6B6B; padding: 3px 10px; border-radius: 20px; }
  .badge-nurture { background: #4ECB7122; color: #4ECB71; border: 1px solid #4ECB71; padding: 3px 10px; border-radius: 20px; }
  .badge-accept  { background: #FFD16622; color: #FFD166; border: 1px solid #FFD166; padding: 3px 10px; border-radius: 20px; }
  .badge-monitor { background: #48CAE422; color: #48CAE4; border: 1px solid #48CAE4; padding: 3px 10px; border-radius: 20px; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ─────────────────────────────────────────────────────────────
data = load_artifacts()
clv_df  = data["clv_df"]
feat_df = data["feat_df"]
prof_df = data["prof_df"]
shap_df = data["shap_df"]
model   = data["model"]

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔭 ChurnLens")
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

contract_options = ["All"] + sorted(clv_df["Contract"].dropna().unique().tolist()) if "Contract" in clv_df.columns else ["All"]
selected_contract = st.sidebar.selectbox("Contract Type", contract_options, key="sidebar_contract")

tenure_range = st.sidebar.slider(
    "Tenure (months)", min_value=0, max_value=72, value=(0, 72), key="sidebar_tenure"
)

quadrant_options = ["All"] + ["Save", "Nurture", "Accept", "Monitor"]
selected_quadrant = st.sidebar.selectbox("Priority Quadrant", quadrant_options, key="sidebar_quadrant")

# Apply filters
mask = pd.Series([True] * len(clv_df), index=clv_df.index)
if selected_contract != "All" and "Contract" in clv_df.columns:
    mask &= clv_df["Contract"] == selected_contract
if "tenure" in clv_df.columns:
    mask &= clv_df["tenure"].between(*tenure_range)
if selected_quadrant != "All":
    mask &= clv_df["priority_quadrant"] == selected_quadrant

filtered_df = clv_df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Customers", f"{len(filtered_df):,}")
st.sidebar.metric("Avg Churn Probability",
    f"{filtered_df['churn_probability'].mean():.1%}" if len(filtered_df) > 0 else "—")
st.sidebar.metric("Total Revenue at Risk",
    f"${filtered_df['revenue_at_risk'].sum():,.0f}" if len(filtered_df) > 0 else "—")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🗂 RFM Segments",
    "🔵 Cluster Profiles",
    "🎯 Churn Model",
    "💰 CLV Analysis",
    "⚡ Action Center",
])


# ════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## ChurnLens — Retention Intelligence Dashboard")
    st.markdown("Real-time customer churn risk analysis powered by XGBoost + SHAP")

    n_customers = len(filtered_df)
    churn_rate  = filtered_df["churn_probability"].mean() if n_customers > 0 else 0
    total_rar   = filtered_df["revenue_at_risk"].sum() if n_customers > 0 else 0
    n_save      = (filtered_df["priority_quadrant"] == "Save").sum() if n_customers > 0 else 0

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <p class="kpi-value" style="color:#6C63FF">{n_customers:,}</p>
            <p class="kpi-label">Total Customers</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
            <p class="kpi-value" style="color:#FF6B6B">{churn_rate:.1%}</p>
            <p class="kpi-label">Avg Churn Probability</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <p class="kpi-value" style="color:#FFD166">${total_rar:,.0f}</p>
            <p class="kpi-label">Revenue at Risk</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <p class="kpi-value" style="color:#FF6B6B">{n_save:,}</p>
            <p class="kpi-label">🚨 Customers to Save</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(churn_distribution_fig(filtered_df), use_container_width=True)
    with col_right:
        st.plotly_chart(priority_matrix_fig(filtered_df), use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — RFM Segments
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## RFM Segments")

    seg_col = "segment" if "segment" in filtered_df.columns else None
    if seg_col is None:
        st.warning("Segment column not found in data. Ensure Day 3 pipeline has run.")
    else:
        fig_seg = rfm_segment_fig(filtered_df)
        if fig_seg:
            st.plotly_chart(fig_seg, use_container_width=True)

        seg_stats = (filtered_df.groupby("segment").agg(
            Customers=("customerID", "count"),
            Avg_Churn_Prob=("churn_probability", "mean"),
            Avg_CLV=("clv_dcf", "mean"),
            Total_RAR=("revenue_at_risk", "sum"),
        ).round(3).reset_index()
        .rename(columns={"segment": "Segment", "Avg_Churn_Prob": "Avg Churn %",
                          "Avg_CLV": "Avg CLV ($)", "Total_RAR": "Total RAR ($)"})
        )
        st.markdown('<p class="section-header">Segment Summary Table</p>', unsafe_allow_html=True)
        st.dataframe(seg_stats.style.format({
            "Avg Churn %": "{:.1%}", "Avg CLV ($)": "${:,.0f}", "Total RAR ($)": "${:,.0f}"
        }), use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — Cluster Profiles
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## K-Means Cluster Profiles")

    cluster_col = "cluster_id" if "cluster_id" in filtered_df.columns else None
    if cluster_col is None:
        st.warning("cluster_id column not found. Ensure Day 3 segmentation pipeline has run.")
    else:
        cluster_stats = (filtered_df.groupby("cluster_id").agg(
            Customers=("customerID", "count"),
            Avg_Churn=("churn_probability", "mean"),
            Avg_CLV=("clv_dcf", "mean"),
            Avg_RAR=("revenue_at_risk", "mean"),
        ).round(3).reset_index()
        .rename(columns={"cluster_id": "Cluster", "Avg_Churn": "Avg Churn %",
                          "Avg_CLV": "Avg CLV ($)", "Avg_RAR": "Avg RAR ($)"})
        )
        st.dataframe(cluster_stats.style.format({
            "Avg Churn %": "{:.1%}", "Avg CLV ($)": "${:,.0f}", "Avg RAR ($)": "${:,.0f}"
        }), use_container_width=True)

        fig_cluster = px.bar(cluster_stats, x="Cluster", y="Avg Churn %",
            color="Avg CLV ($)", color_continuous_scale="viridis",
            title="Cluster Average Churn Probability",
            labels={"Cluster": "Cluster ID", "Avg Churn %": "Avg Churn Probability"})
        fig_cluster.update_layout(
            paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["bg"],
            font_color=PALETTE["text"], title_font_color=PALETTE["text"],
        )
        st.plotly_chart(fig_cluster, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 4 — Churn Model
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Churn Model Evaluation")

    # Live threshold slider
    threshold = st.slider(
        "Decision Threshold (move to balance precision vs recall)",
        min_value=0.05, max_value=0.95, value=0.50, step=0.01,
        key="churn_threshold_slider",
        help="Lower threshold → catch more churners (higher recall, lower precision)."
    )

    # Build predictions from the clv_df (already has churn_probability)
    y_proba = filtered_df["churn_probability"].values
    y_true  = (filtered_df["churn_probability"] >= 0.5).astype(int).values  # proxy if no ground truth
    # Use Churn column from feat_df if available
    feat_merged = filtered_df.merge(
        feat_df[["customerID", "Churn"]], on="customerID", how="left"
    ) if "customerID" in filtered_df.columns and "customerID" in feat_df.columns else None
    if feat_merged is not None and "Churn" in feat_merged.columns:
        y_true = feat_merged["Churn"].values.astype(int)

    y_pred = (y_proba >= threshold).astype(int)
    from sklearn.metrics import confusion_matrix as sk_cm, f1_score, precision_score, recall_score
    try:
        tn, fp, fn, tp = sk_cm(y_true, y_pred, labels=[0, 1]).ravel()
        prec  = precision_score(y_true, y_pred, zero_division=0)
        rec   = recall_score(y_true, y_pred, zero_division=0)
        f1    = f1_score(y_true, y_pred, zero_division=0)
    except Exception:
        tn = fp = fn = tp = 0
        prec = rec = f1 = 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{prec:.3f}")
    m2.metric("Recall",    f"{rec:.3f}")
    m3.metric("F1 Score",  f"{f1:.3f}")
    m4.metric("Threshold", f"{threshold:.2f}")

    col_cm, col_pr = st.columns(2)
    with col_cm:
        st.plotly_chart(confusion_matrix_fig(tn, fp, fn, tp, threshold), use_container_width=True)
    with col_pr:
        if len(y_proba) > 1:
            pr_result = compute_pr_curve(y_true, y_proba)
            st.plotly_chart(pr_curve_fig(pr_result["pr_df"], pr_result["pr_auc"]), use_container_width=True)

    # Calibration
    if len(y_proba) > 1:
        cal_result = compute_calibration(y_true, y_proba)
        col_cal, col_info = st.columns([2, 1])
        with col_cal:
            st.plotly_chart(calibration_fig(cal_result["calibration_df"], cal_result["ece"]), use_container_width=True)
        with col_info:
            st.markdown('<p class="section-header">Calibration Health</p>', unsafe_allow_html=True)
            ece_val = cal_result["ece"]
            ece_color = "🟢" if ece_val <= 0.05 else "🟡" if ece_val <= 0.10 else "🔴"
            st.markdown(f"**ECE:** {ece_val:.4f} {ece_color}")
            st.markdown("""
            **ECE ≤ 0.05** → Excellent calibration.
            Predicted probabilities match real-world churn frequencies.
            Revenue-at-risk calculations are financially reliable.
            """)

    # SHAP global beeswarm
    st.markdown("---")
    st.markdown('<p class="section-header">Global Feature Importance (SHAP)</p>', unsafe_allow_html=True)
    top_n = st.slider("Number of features to display", 5, 26, 15, key="shap_top_n")
    st.plotly_chart(shap_beeswarm_fig(shap_df, top_n=top_n), use_container_width=True)

    # SHAP dependence plots
    st.markdown('<p class="section-header">SHAP Dependence Plots — Top 3 Features</p>', unsafe_allow_html=True)
    feat_names = list(shap_df.columns)
    importance  = shap_df.abs().mean().sort_values(ascending=False)
    top3_feats  = importance.head(3).index.tolist()
    dep_cols    = st.columns(3)
    for i, feat in enumerate(top3_feats):
        feat_idx = feat_names.index(feat)
        dep_data = pd.DataFrame({
            "feature_name":  feat,
            "feature_value": shap_df.index,   # just indices as proxy (full X not stored)
            "shap_value":    shap_df[feat].values,
        })
        with dep_cols[i]:
            st.plotly_chart(dependence_fig(dep_data), use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 5 — CLV Analysis
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## Customer Lifetime Value Analysis")

    col_h, col_r = st.columns(2)
    with col_h:
        st.plotly_chart(clv_heatmap_fig(filtered_df), use_container_width=True)
    with col_r:
        fig_rar = px.histogram(filtered_df, x="revenue_at_risk", nbins=40,
            title="Revenue at Risk Distribution",
            color_discrete_sequence=[PALETTE["warning"]],
            labels={"revenue_at_risk": "Revenue at Risk ($)"})
        fig_rar.update_layout(
            paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["bg"],
            font_color=PALETTE["text"], title_font_color=PALETTE["text"],
        )
        st.plotly_chart(fig_rar, use_container_width=True)

    fig_clv_scatter = px.scatter(
        filtered_df.sample(min(2000, len(filtered_df)), random_state=42),
        x="churn_probability", y="clv_dcf",
        color="priority_quadrant",
        color_discrete_map={"Save":"#FF6B6B","Nurture":"#4ECB71","Accept":"#FFD166","Monitor":"#48CAE4"},
        size="revenue_at_risk",
        hover_data=["customerID"] if "customerID" in filtered_df.columns else None,
        title="CLV vs Churn Risk — Priority Quadrants",
        labels={"churn_probability": "Churn Probability", "clv_dcf": "CLV — DCF ($)"},
        opacity=0.7,
    )
    fig_clv_scatter.update_layout(
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["bg"],
        font_color=PALETTE["text"], title_font_color=PALETTE["text"],
    )
    st.plotly_chart(fig_clv_scatter, use_container_width=True)

    # Summary table
    st.markdown('<p class="section-header">CLV Bin Summary</p>', unsafe_allow_html=True)
    clv_summary = (filtered_df.groupby("clv_bin").agg(
        Customers=("customerID", "count"),
        Avg_CLV=("clv_dcf", "mean"),
        Avg_Churn=("churn_probability", "mean"),
        Total_RAR=("revenue_at_risk", "sum"),
    ).round(2).reset_index()
    .rename(columns={"clv_bin":"CLV Tier","Avg_CLV":"Avg CLV ($)",
                      "Avg_Churn":"Avg Churn %","Total_RAR":"Total RAR ($)"})
    )
    st.dataframe(clv_summary.style.format({
        "Avg CLV ($)":"${:,.0f}", "Avg Churn %":"{:.1%}", "Total RAR ($)":"${:,.0f}"
    }), use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 6 — Action Center
# ════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## ⚡ Action Center — Per-Customer Retention Intelligence")

    if "customerID" not in filtered_df.columns:
        st.error("customerID column not available in filtered data.")
    else:
        customer_ids = sorted(filtered_df["customerID"].dropna().unique().tolist())
        selected_id  = st.selectbox(
            "🔍 Select Customer ID", customer_ids, key="action_customer_select"
        )

        row = filtered_df[filtered_df["customerID"] == selected_id]
        if row.empty:
            st.warning("Customer not found in filtered data. Adjust sidebar filters.")
        else:
            row = row.iloc[0]
            churn_prob = float(row.get("churn_probability", 0))
            clv_val    = float(row.get("clv_dcf", 0))
            rar_val    = float(row.get("revenue_at_risk", 0))
            quadrant   = str(row.get("priority_quadrant", "—"))
            tenure_val = float(row.get("tenure", 0)) if "tenure" in row.index else 0

            # Top metrics
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Churn Probability", f"{churn_prob:.1%}")
            a2.metric("Customer CLV",      f"${clv_val:,.0f}")
            a3.metric("Revenue at Risk",   f"${rar_val:,.0f}")
            a4.metric("Tenure",            f"{int(tenure_val)} months")

            col_gauge, col_action = st.columns([1, 2])
            with col_gauge:
                st.plotly_chart(churn_gauge_fig(churn_prob, selected_id), use_container_width=True)
            with col_action:
                badge_map = {
                    "Save":    ("badge-save",    "🚨 SAVE",    "Immediate retention outreach. High CLV + High Risk. Offer contract upgrade or loyalty discount."),
                    "Nurture": ("badge-nurture", "🌱 NURTURE", "Growth opportunity. High CLV + Low Risk. Focus on upselling premium services."),
                    "Accept":  ("badge-accept",  "⚠️ ACCEPT",  "Low CLV + High Risk. Minimal intervention. Cost of retention likely exceeds revenue saved."),
                    "Monitor": ("badge-monitor", "👁 MONITOR", "Low CLV + Low Risk. Routine check-in. No urgent action needed."),
                }
                badge_cls, badge_label, rec_text = badge_map.get(
                    quadrant, ("badge-monitor", quadrant, "Monitor customer health.")
                )
                st.markdown(f'<span class="{badge_cls}">{badge_label}</span>', unsafe_allow_html=True)
                st.markdown(f"**Priority Quadrant:** {quadrant}")
                st.markdown(f"**Recommended Action:**  \n{rec_text}")

                st.markdown("---")
                # Customer profile summary
                profile_cols = ["Contract", "InternetService", "tenure",
                                "MonthlyCharges", "TotalCharges"]
                profile_cols = [c for c in profile_cols if c in row.index]
                if profile_cols:
                    st.markdown('<p class="section-header">Customer Profile</p>', unsafe_allow_html=True)
                    for c in profile_cols:
                        val = row[c]
                        if isinstance(val, float):
                            st.markdown(f"- **{c}:** {val:,.2f}")
                        else:
                            st.markdown(f"- **{c}:** {val}")

            # SHAP waterfall
            st.markdown('<p class="section-header">SHAP Feature Contributions</p>', unsafe_allow_html=True)
            if len(shap_df) > 0:
                shap_idx = 0  # default: first sample
                # Try to find customer in SHAP sample if customerID stored
                if "customerID" in shap_df.columns:
                    match = shap_df[shap_df["customerID"] == selected_id]
                    if not match.empty:
                        shap_idx = match.index[0]
                        shap_row_data = match.drop(columns=["customerID"]).iloc[0]
                    else:
                        shap_row_data = shap_df.drop(columns=["customerID"], errors="ignore").iloc[shap_idx]
                else:
                    shap_row_data = shap_df.iloc[shap_idx]

                st.plotly_chart(
                    shap_waterfall_fig(shap_row_data, selected_id),
                    use_container_width=True,
                )
            else:
                st.info("SHAP values not available. Run the Day 6 pipeline first.")
