"""dashboard/charts.py — reusable Plotly chart builders."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


PALETTE = {
    "primary":   "#6C63FF",
    "secondary": "#48CAE4",
    "danger":    "#FF6B6B",
    "success":   "#4ECB71",
    "warning":   "#FFD166",
    "bg":        "#0F1117",
    "card":      "#1E2130",
    "text":      "#E0E0E0",
}

QUADRANT_COLORS = {
    "Save":    "#FF6B6B",
    "Nurture": "#4ECB71",
    "Accept":  "#FFD166",
    "Monitor": "#48CAE4",
}


def overview_kpi_fig(n_customers, churn_rate, total_rar, top_feature):
    """Four KPI cards as a single-row Indicator figure."""
    fig = go.Figure()
    fig.add_trace(go.Indicator(mode="number", value=n_customers,
        title={"text": "Total Customers", "font": {"color": PALETTE["text"]}},
        number={"font": {"color": PALETTE["primary"]}},
        domain={"x": [0, 0.25], "y": [0, 1]}))
    fig.add_trace(go.Indicator(mode="number", value=round(churn_rate * 100, 1),
        title={"text": "Churn Rate %", "font": {"color": PALETTE["text"]}},
        number={"font": {"color": PALETTE["danger"]}, "suffix": "%"},
        domain={"x": [0.25, 0.5], "y": [0, 1]}))
    fig.add_trace(go.Indicator(mode="number", value=round(total_rar / 1_000, 1),
        title={"text": "Revenue at Risk ($K)", "font": {"color": PALETTE["text"]}},
        number={"font": {"color": PALETTE["warning"]}, "suffix": "K"},
        domain={"x": [0.5, 0.75], "y": [0, 1]}))
    fig.add_trace(go.Indicator(mode="number+delta", value=0,
        title={"text": f"Top Risk Driver: {top_feature}", "font": {"color": PALETTE["text"]}},
        number={"font": {"color": PALETTE["secondary"]}},
        domain={"x": [0.75, 1.0], "y": [0, 1]}))
    fig.update_layout(
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["card"],
        height=160, margin=dict(t=20, b=10, l=10, r=10),
    )
    return fig


def churn_distribution_fig(clv_df: pd.DataFrame):
    fig = px.histogram(
        clv_df, x="churn_probability", nbins=40,
        color_discrete_sequence=[PALETTE["primary"]],
        labels={"churn_probability": "Churn Probability"},
        title="Churn Probability Distribution",
    )
    fig.update_layout(
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["bg"],
        font_color=PALETTE["text"], title_font_color=PALETTE["text"],
        margin=dict(t=50, b=30, l=30, r=20),
    )
    return fig


def priority_matrix_fig(clv_df: pd.DataFrame):
    counts = clv_df["priority_quadrant"].value_counts().reset_index()
    counts.columns = ["quadrant", "count"]
    colors = [QUADRANT_COLORS.get(q, "#aaa") for q in counts["quadrant"]]
    fig = px.bar(counts, x="quadrant", y="count", color="quadrant",
        color_discrete_map=QUADRANT_COLORS,
        title="2×2 Retention Priority Matrix",
        labels={"quadrant": "Priority Quadrant", "count": "# Customers"})
    fig.update_layout(
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["bg"],
        font_color=PALETTE["text"], title_font_color=PALETTE["text"],
        showlegend=False, margin=dict(t=50, b=30, l=30, r=20),
    )
    return fig


def clv_heatmap_fig(clv_df: pd.DataFrame):
    pivot = (clv_df.groupby(["clv_bin", "churn_risk_bin"])["revenue_at_risk"]
             .mean().unstack(fill_value=0))
    bin_order = ["Low", "Medium", "High"]
    pivot = pivot.reindex(index=[b for b in bin_order if b in pivot.index],
                          columns=[b for b in bin_order if b in pivot.columns])
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        colorscale="RdYlGn_r",
        text=np.round(pivot.values, 0),
        texttemplate="$%{text:,.0f}",
        colorbar=dict(title="Avg Revenue<br>at Risk ($)"),
    ))
    fig.update_layout(
        title="CLV × Churn Risk Heatmap (Avg Revenue at Risk)",
        xaxis_title="Churn Risk Bin", yaxis_title="CLV Bin",
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["bg"],
        font_color=PALETTE["text"], title_font_color=PALETTE["text"],
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def shap_beeswarm_fig(shap_df: pd.DataFrame, top_n: int = 15):
    importance = shap_df.abs().mean().sort_values(ascending=False).head(top_n)
    fig = go.Figure(go.Bar(
        x=importance.values[::-1], y=importance.index[::-1],
        orientation="h",
        marker=dict(
            color=importance.values[::-1],
            colorscale="Viridis", showscale=True,
            colorbar=dict(title="Mean |SHAP|"),
        ),
    ))
    fig.update_layout(
        title=f"Global Feature Importance — Top {top_n} Features (Mean |SHAP|)",
        xaxis_title="Mean |SHAP value|", yaxis_title="",
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["bg"],
        font_color=PALETTE["text"], title_font_color=PALETTE["text"],
        height=max(400, top_n * 28), margin=dict(t=50, b=30, l=180, r=20),
    )
    return fig


def shap_waterfall_fig(shap_row: pd.Series, customer_id: str, base_value: float = 0.0):
    sv = shap_row.sort_values(key=abs, ascending=False).head(12)
    colors = [PALETTE["danger"] if v > 0 else PALETTE["success"] for v in sv.values]
    fig = go.Figure(go.Bar(
        x=sv.values, y=sv.index, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in sv.values],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"SHAP Waterfall — Customer {customer_id}",
        xaxis_title="SHAP Contribution (→ higher churn risk)",
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["bg"],
        font_color=PALETTE["text"], title_font_color=PALETTE["text"],
        height=420, margin=dict(t=50, b=30, l=180, r=80),
    )
    return fig


def pr_curve_fig(pr_df: pd.DataFrame, pr_auc: float):
    fig = px.line(pr_df, x="recall", y="precision",
        title=f"Precision-Recall Curve (AUC = {pr_auc:.3f})",
        color_discrete_sequence=[PALETTE["primary"]])
    fig.add_hline(y=0.26, line_dash="dash", line_color=PALETTE["warning"],
        annotation_text="Random baseline (26% churn rate)")
    fig.update_layout(
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["bg"],
        font_color=PALETTE["text"], title_font_color=PALETTE["text"],
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def calibration_fig(cal_df: pd.DataFrame, ece: float):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color=PALETTE["warning"]), name="Perfect calibration"))
    fig.add_trace(go.Scatter(
        x=cal_df["mean_predicted_value"], y=cal_df["fraction_of_positives"],
        mode="lines+markers", name="Model",
        line=dict(color=PALETTE["primary"]),
        marker=dict(size=8),
    ))
    fig.update_layout(
        title=f"Calibration Curve (ECE = {ece:.4f})",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["bg"],
        font_color=PALETTE["text"], title_font_color=PALETTE["text"],
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def confusion_matrix_fig(tn, fp, fn, tp, threshold):
    z = [[tn, fp], [fn, tp]]
    text = [[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicted: No Churn", "Predicted: Churn"],
        y=["Actual: No Churn", "Actual: Churn"],
        text=text, texttemplate="%{text}",
        colorscale=[[0, PALETTE["success"]], [1, PALETTE["danger"]]],
        showscale=False,
    ))
    fig.update_layout(
        title=f"Confusion Matrix @ threshold = {threshold:.3f}",
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["bg"],
        font_color=PALETTE["text"], title_font_color=PALETTE["text"],
        height=350, margin=dict(t=50, b=50, l=120, r=20),
    )
    return fig


def dependence_fig(dep_df: pd.DataFrame):
    feat = dep_df["feature_name"].iloc[0]
    fig = px.scatter(dep_df, x="feature_value", y="shap_value",
        title=f"SHAP Dependence — {feat}",
        labels={"feature_value": feat, "shap_value": "SHAP Value"},
        color="shap_value", color_continuous_scale="RdBu_r",
        opacity=0.7)
    fig.update_layout(
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["bg"],
        font_color=PALETTE["text"], title_font_color=PALETTE["text"],
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def churn_gauge_fig(churn_prob: float, customer_id: str):
    color = (PALETTE["danger"] if churn_prob >= 0.5
             else PALETTE["warning"] if churn_prob >= 0.3
             else PALETTE["success"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(churn_prob * 100, 1),
        number={"suffix": "%", "font": {"color": color, "size": 36}},
        title={"text": f"Churn Risk<br>{customer_id}", "font": {"color": PALETTE["text"]}},
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=PALETTE["text"]),
            bar=dict(color=color),
            bgcolor=PALETTE["card"],
            steps=[
                dict(range=[0, 30],  color="#1a2a1a"),
                dict(range=[30, 60], color="#2a2a1a"),
                dict(range=[60, 100], color="#2a1a1a"),
            ],
            threshold=dict(line=dict(color="white", width=3), value=50),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=PALETTE["card"], height=280,
        margin=dict(t=60, b=20, l=30, r=30),
    )
    return fig


def rfm_segment_fig(clv_df: pd.DataFrame):
    if "segment" not in clv_df.columns:
        return None
    seg_stats = clv_df.groupby("segment").agg(
        count=("customerID", "count"),
        avg_clv=("clv_dcf", "mean"),
        avg_churn=("churn_probability", "mean"),
    ).reset_index()
    fig = px.scatter(seg_stats, x="avg_churn", y="avg_clv",
        size="count", color="segment", text="segment",
        title="RFM Segments — Churn Risk vs CLV",
        labels={"avg_churn": "Avg Churn Probability", "avg_clv": "Avg CLV ($)"},
        size_max=60)
    fig.update_traces(textposition="top center")
    fig.update_layout(
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["bg"],
        font_color=PALETTE["text"], title_font_color=PALETTE["text"],
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig
