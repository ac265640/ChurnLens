# ChurnLens

**Production Customer Churn & Retention System**

ChurnLens is an end-to-end data science system that ingests raw telecom customer data and produces:
- Behavioral segmentation (RFM + K-Means)
- Calibrated churn probability (XGBoost + Optuna + SMOTE)
- Customer Lifetime Value estimates (BG/NBD + DCF)
- SHAP-based model interpretation (global + local)

It culminates in a deployed Streamlit dashboard with a live threshold slider, CLV×churn heatmap, and per-customer SHAP waterfall.

## Tech  Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, Optuna, imbalanced-learn
- SHAP, Lifetimes
- Streamlit, Plotly
- pytest, Git

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ac265640/ChurnLens.git
   cd ChurnLens
   ```

2. **Set up the virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   ```bash
   python scripts/download_data.py
   ```

5. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

## Dataset
Uses the Telco Customer Churn dataset (IBM Watson / Kaggle), containing ~7,000 rows, 21 features, and a 26% churn rate.
