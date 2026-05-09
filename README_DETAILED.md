# RiskInsight XGB - Detailed Project Documentation

This file explains the project in simple words, but with full detail.

---

## 1. Project objective

The goal is to predict **credit default risk** for one applicant at a time, and also explain the reason behind the prediction.

In short:
- Take borrower financial profile as input
- Predict probability of default
- Convert probability into decision (high risk / low risk) using a threshold
- Explain what features increased or reduced risk

---

## 2. What this system does end-to-end

1. User opens Streamlit app UI.
2. User enters borrower details (age, income, debt metrics, late payment counts, etc.).
3. App builds a feature vector in tabular format.
4. App loads trained XGBoost model and feature order artifact.
5. App aligns columns exactly in training order.
6. App predicts probability (`predict_proba`).
7. App compares probability with user-selected threshold.
8. App renders:
   - probability gauge
   - final decision badge
   - SHAP impact chart for explainability
9. App shows raw inference vector in expander for transparency.

---

## 3. Repository structure and purpose of each file

```text
.
├── app.py
├── README.md
├── README_DETAILED.md
├── requirements.txt
├── .gitignore
└── artifacts
    ├── model-RiskInsight-XGB.pkl
    └── features-RiskInsight-XGB.pkl
```

- `app.py`  
  Main Streamlit application: UI, prediction pipeline, visualizations, and SHAP explanation.

- `artifacts/model-RiskInsight-XGB.pkl`  
  Pre-trained XGBoost classifier used at inference time.

- `artifacts/features-RiskInsight-XGB.pkl`  
  Exact feature list/order used during training. This prevents wrong-column prediction errors.

- `requirements.txt`  
  Python dependencies required to run the app.

- `.gitignore`  
  Prevents committing cache/env/log files.

- `README.md`  
  Main quick-start documentation.

- `README_DETAILED.md`  
  This detailed technical explanation file.

---

## 4. Application pipeline in detail (`app.py`)

### 4.1 UI and configuration layer

- Streamlit page is configured with title/icon/layout.
- Custom CSS is injected for dark dashboard style, panels, and visuals.
- UI has three panels:
  - Demographics & Financials
  - Credit & Payment History
  - System Controls (decision threshold)

### 4.2 Artifact loading layer

`load_artifacts()` does:
- Load model from `artifacts/model-RiskInsight-XGB.pkl`
- Load ordered feature list from `artifacts/features-RiskInsight-XGB.pkl`
- Cache both using `@st.cache_resource` so reload is fast

### 4.3 Input collection layer

The app collects:
- `Age`
- `Dependents`
- `CreditUtilization`
- `DebtRatio`
- `MonthlyIncome`
- `OpenCreditLines`
- `RealEstateLoans`
- `Late30_59`
- `Late60_89`
- `Late90Plus`
- `TotalPastDue`

### 4.4 Inference preprocessing layer

Before prediction:
1. Input values are packed into a pandas DataFrame (single row).
2. Columns are reordered to `features` artifact order:
   `input_data = input_data[features]`

This is a critical safety step so the model sees the exact same feature order as training.

### 4.5 Prediction layer

- Model call: `model.predict_proba(input_data)[0][1]`
- Output is probability of default (`prob`).
- User-defined threshold is applied:
  - if `prob > threshold` -> high risk
  - else -> low risk/approved

### 4.6 Explainability layer (SHAP)

The app computes local explanation for the current row:
- `explainer = shap.Explainer(model)`
- `shap_values = explainer(input_data)`
- Extracts per-feature SHAP impacts
- Ranks by absolute impact
- Shows top 8 in horizontal bar chart

Color meaning:
- Red -> pushes risk up
- Green -> pushes risk down

### 4.7 Visualization layer

- Plotly gauge: probability + threshold marker
- Verdict badge with dynamic color
- SHAP impact chart
- Raw inference table expander

---

## 5. Preprocessing details

### 5.1 Input-level preprocessing (implemented)

- Numeric input constraints are enforced by Streamlit controls:
  - Age: 18-100
  - Dependents: 0-10
  - CreditUtilization: 0.0-1.0
  - DebtRatio: 0.0-5.0
  - MonthlyIncome: 0-1,000,000
  - OpenCreditLines: 0-20
  - RealEstateLoans: 0-10
  - Late payment buckets: bounded integer ranges
  - TotalPastDue: 0-20
- Missing values are avoided because all fields have defaults.
- Final column order is aligned using saved feature list artifact.

### 5.2 Model-side preprocessing

No scaler/normalizer is applied inside `app.py`.  
The app assumes the saved model is already trained to consume raw numeric features in this format.

---

## 6. Feature engineering details

There is no additional runtime feature engineering step in `app.py` beyond constructing the inference row.

However, the model uses **already engineered risk indicators** such as:
- `CreditUtilization` (ratio-style risk feature)
- `DebtRatio` (ratio-style risk feature)
- `TotalPastDue` (aggregated delinquency signal)

So, the app uses engineered-style features as model inputs, even though feature engineering code is not part of this repository.

---

## 7. Model details

Loaded model type:
- `xgboost.sklearn.XGBClassifier`

Key parameters found in saved artifact:
- `objective = binary:logistic`
- `n_estimators = 200`
- `max_depth = 4`
- `learning_rate = 0.05`
- `subsample = 1.0`
- `colsample_bytree = 0.7`
- `random_state = 42`

This is a binary classification setup for default risk prediction.

---

## 8. Feature list used by model (exact order)

1. `CreditUtilization`
2. `Age`
3. `Late30_59`
4. `DebtRatio`
5. `MonthlyIncome`
6. `OpenCreditLines`
7. `Late90Plus`
8. `RealEstateLoans`
9. `Late60_89`
10. `Dependents`
11. `TotalPastDue`

---

## 9. Tech stack and role of each dependency

- `streamlit` -> Web app framework
- `pandas` -> DataFrame creation/manipulation
- `numpy` -> Numeric utility operations
- `plotly` -> Gauge and SHAP impact charts
- `joblib` -> Load model and feature artifacts
- `shap` -> Local explainability
- `xgboost` -> ML model runtime
- `scikit-learn` -> API compatibility + ecosystem support

---

## 10. What was done in this repository setup

1. Organized artifacts under `artifacts/` folder.
2. Updated `app.py` to load model/features from artifact directory using robust path logic.
3. Added dependency file (`requirements.txt`).
4. Added `.gitignore` for clean repository.
5. Added documentation:
   - quick `README.md`
   - this detailed `README_DETAILED.md`
6. Pushed the structured repository to GitHub main branch.

---

## 11. How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in terminal (usually `http://localhost:8501`).

---

## 12. How to read model output correctly

- Probability is not yes/no by itself. It is risk confidence.
- Threshold is policy control:
  - lower threshold -> stricter decisions (more high-risk flags)
  - higher threshold -> lenient decisions
- SHAP explains one specific prediction, not global dataset behavior.

---

## 13. Current limitations

- Training notebook/script is not included in this repo.
- No dataset is bundled here.
- No model monitoring/calibration pipeline included yet.
- No batch scoring endpoint; current app is interactive single-case scoring.

---

## 14. Suggested next upgrades (optional roadmap)

- Add training pipeline scripts/notebooks (`data prep -> train -> evaluate -> export`).
- Add metrics section (AUC, precision, recall, confusion matrix).
- Add threshold analytics and policy simulation charts.
- Add API endpoint for batch inference.
- Add model versioning and experiment tracking.

---

## 15. Author

- **Singhrituraj114**
