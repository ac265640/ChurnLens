import pandas as pd
import numpy as np
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

# 1. Load artifacts
with open('data/artifacts/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Construct feature vector based on user description
row = {
    'SeniorCitizen': 0,
    'Partner': 0,
    'Dependents': 0,
    'tenure': 2,
    'PhoneService': 1,
    'MultipleLines': 0,
    'OnlineSecurity': 0,
    'OnlineBackup': 0,
    'DeviceProtection': 0,
    'TechSupport': 0,
    'StreamingTV': 1,
    'StreamingMovies': 0,
    'Contract': 0,  # Month-to-month
    'PaperlessBilling': 1,
    'MonthlyCharges': 85.0,
    'TotalCharges': 170.0,
    'Churn': 0,
    'rfm_recency': 2.0,
    'rfm_frequency': 3.0,
    'rfm_monetary': 85.0,
    'tenure_decay': np.exp(-0.04 * 2),
    'gender_1': 1,
    'InternetService_Fiber_optic': 1,
    'InternetService_No': 0,
    'PaymentMethod_Credit_card_(automatic)': 0,
    'PaymentMethod_Electronic_check': 1,
    'PaymentMethod_Mailed_check': 0
}

df = pd.DataFrame([row])

feat_df = pd.read_parquet('data/processed/features.parquet')
cols = [c for c in feat_df.columns if c not in ['Churn','customerID','segment','cluster_id','R_score','F_score','M_score','rfm_score'] and feat_df[c].dtype != object]
X = df[cols].values.astype(float)

# 3. Predict Churn
churn_prob = model.predict_proba(X)[0, 1]

# 4. Predict CLV
from src.models.clv import compute_dcf_clv, expected_tenure_from_churn_prob, assign_priority_quadrant
expected_tenure = expected_tenure_from_churn_prob(churn_prob)
clv = compute_dcf_clv(85.0, expected_tenure)

clv_df = pd.read_parquet('data/processed/customer_clv.parquet')
median_clv = clv_df['clv_dcf'].median()
quadrant = assign_priority_quadrant([clv], [churn_prob], clv_threshold=median_clv, churn_threshold=0.30)[0]

# 5. SHAP values
from src.models.shap_explainer import compute_shap_values
shap_vals, explainer, idx = compute_shap_values(model, X, cols, max_samples=1)
sv = shap_vals[0]
shap_series = pd.Series(sv, index=cols).sort_values(key=abs, ascending=False).head(4)

print(f"--- CUSTOMER ANALYSIS ---")
print(f"Churn Probability: {churn_prob:.1%}")
print(f"Expected Tenure:   {expected_tenure:.1f} months")
print(f"Estimated CLV:     ${clv:,.0f}")
print(f"Quadrant:          {quadrant}")
print(f"Top 4 Risk Drivers (SHAP):")
for k, v in shap_series.items():
    direction = "Increases Risk" if v > 0 else "Decreases Risk"
    print(f"  - {k}: {v:+.3f} ({direction})")
