# ChurnLens — Retention Intelligence Dashboard

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-churnlenss.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit)](https://churnlenss.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-blue?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **A production-grade customer churn prediction and retention system** built on the IBM Telco Churn dataset (~7,000 customers). Ingests raw CSV data and delivers a fully interactive business intelligence dashboard — no manual steps required.

---

## 🔴 Live Demo

**👉 [https://churnlenss.streamlit.app/](https://churnlenss.streamlit.app/)**

---

## 📊 What ChurnLens Does

ChurnLens transforms raw telecom customer records into actionable retention intelligence through a complete ML pipeline:

| Layer | What it produces |
|---|---|
| **Feature Engineering** | RFM  scores, temporal decay, service frequency, one-hot encoding |
| **Segmentation** | 4 named behavioural segments via RFM scoring + K-Means clustering |
| **Churn Model** | Calibrated XGBoost classifier (SMOTE-in-CV, Optuna HPO, Platt scaling) |
| **CLV Engine** | Discounted Cash Flow + BG/NBD lifetime value per customer |
| **Explainability** | SHAP global feature importance + per-customer waterfall charts |
| **Dashboard** | 6-tab Streamlit app with live filters, threshold slider, and action centre |

---

## 🗂 Dashboard Tabs

| Tab | Description |
|---|---|
| 📊 **Overview** | KPI cards (total customers, avg churn risk, revenue at risk, customers to save) + 2×2 retention priority matrix |
| 🗂 **RFM Segments** | Scatter plot and summary table for Champions, Loyal Active, At-Risk, Hibernating |
| 🌐 **Cluster Profiles** | K-Means cluster behaviour heatmap and radar chart |
| 🤖 **Churn Model** | ROC curve, calibration plot, confusion matrix with live threshold slider |
| 💰 **CLV Analysis** | CLV vs churn probability heatmap, segment-level revenue breakdown |
| ⚡ **Action Center** | Look up any customer by ID → get churn score, CLV, segment, recommended action, and SHAP waterfall |

---

## 🏗 Architecture

```
Raw CSV  →  Preprocessor  →  Feature Matrix
                                 ↓
                       ┌─────────────────────┐
                       │   Segmentation      │  RFM Quartile Scoring
                       │   (K-Means, k=2)    │  + Named Segments
                       └─────────────────────┘
                                 ↓
                       ┌─────────────────────┐
                       │   Churn Model       │  XGBoost + SMOTE + Optuna
                       │   (Calibrated)      │  + Platt Scaling
                       └─────────────────────┘
                                 ↓
                       ┌─────────────────────┐
                       │   CLV Engine        │  DCF + BG/NBD
                       │   Priority Matrix   │  Save / Nurture / Monitor / Accept
                       └─────────────────────┘
                                 ↓
                       ┌─────────────────────┐
                       │   SHAP Explainer    │  TreeExplainer → shap_values.parquet
                       └─────────────────────┘
                                 ↓
                       ┌─────────────────────┐
                       │   Streamlit         │  6-Tab Dashboard (Live)
                       │   Dashboard         │
                       └─────────────────────┘
```

---

## 🛠 Tech Stack

| Category | Libraries |
|---|---|
| Data | `pandas`, `numpy`, `pyarrow` |
| ML | `scikit-learn`, `xgboost`, `imbalanced-learn`, `optuna` |
| CLV | `lifetimes` (BG/NBD) |
| Explainability | `shap` |
| Dashboard | `streamlit`, `plotly` |
| Testing | `pytest`, `pytest-cov` (49 tests across 7 modules) |

---

## 📁 Project Structure

```
ChurnLens/
├── dashboard/
│   ├── app.py          # 6-tab Streamlit dashboard
│   ├── charts.py       # Plotly chart library (gauge, waterfall, heatmap…)
│   └── loader.py       # Cached artifact loader (st.cache_resource)
├── src/
│   ├── data/
│   │   ├── loader.py       # Schema-validated CSV ingestion
│   │   └── preprocessor.py # Full feature engineering pipeline
│   ├── features/
│   │   └── segmentation.py # RFM scoring + K-Means + segment naming
│   └── models/
│       ├── churn_model.py  # XGBoost pipeline + threshold optimisation
│       ├── clv.py          # DCF CLV + priority matrix
│       └── shap_explainer.py # TreeExplainer + persistence
├── data/
│   ├── raw/            # Source Telco CSV
│   ├── processed/      # Engineered features + customer profiles + CLV table
│   └── artifacts/      # xgb_model.pkl, shap_values.parquet
├── tests/              # 49 unit tests (Day 1–6 coverage)
├── scripts/            # Data download utility
└── requirements.txt
```

---

## 🚀 Run Locally

```bash
# 1. Clone
git clone https://github.com/ac265640/ChurnLens.git
cd ChurnLens

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch dashboard
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser. Pre-computed artifacts are included in the repo — no retraining needed.

---

## 🧪 Run Tests

```bash
pytest tests/ -v --tb=short
```

49 tests covering data loading, feature engineering, segmentation, churn model, CLV engine, and SHAP explainer.

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | ~0.85 |
| Calibration | Platt-scaled (CalibratedClassifierCV) |
| Optimal Threshold | 0.14 (cost-minimising: FN penalty 5× FP) |
| Customers Flagged to Save | 332 (High risk + High CLV) |
| Portfolio Revenue at Risk | ~$3,000,000 |

---

## 📋 Dataset

IBM Telco Customer Churn dataset — ~7,043 rows, 21 raw features, 26.5% churn rate.  
Available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
