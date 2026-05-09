# RiskInsight XGB Credit Risk Prediction Explainability Engine

This project is a **credit risk prediction web app**.  
You enter applicant financial details, and the app predicts the chance of default using an XGBoost model.  
It also explains *why* the model predicted that result using SHAP.

## Full detailed documentation

For complete end-to-end details (pipeline, preprocessing, feature engineering, model logic, explainability flow, and what was implemented), read:

- **[README_DETAILED.md](README_DETAILED.md)**

## What this project does

1. Takes borrower inputs (age, income, debt ratio, late payments, etc.).
2. Calculates default probability with a trained ML model.
3. Compares that probability to a threshold you control.
4. Shows decision as:
   - **HIGH RISK DETECTED**
   - **LOW RISK · APPROVED**
5. Explains feature impact (what pushed risk up/down).

## Why this is useful

- Fast risk scoring for decision support.
- Transparent model behavior (not a black box).
- Interactive threshold control for strict/lenient policy simulation.
- Easy demo-ready UI built with Streamlit.

## Project structure

```text
.
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── artifacts
    ├── model-RiskInsight-XGB.pkl
    └── features-RiskInsight-XGB.pkl
```

## End-to-end working (simple flow)

### 1) App starts
- `app.py` launches a Streamlit dashboard with custom dark UI styling.

### 2) Model artifacts load
- `artifacts/model-RiskInsight-XGB.pkl` → trained `XGBClassifier`
- `artifacts/features-RiskInsight-XGB.pkl` → exact feature order used in training

### 3) User enters inputs
- Inputs are collected from sliders and number fields.

### 4) Input is converted to model format
- Values are packed into a pandas DataFrame.
- Columns are reordered using the saved feature list so they exactly match training format.

### 5) Prediction is computed
- `predict_proba()` returns probability of default.
- Probability is compared to threshold.

### 6) Decision and visuals are rendered
- Gauge chart shows probability and threshold.
- Verdict badge shows risk decision.

### 7) SHAP explainability runs
- SHAP computes feature-level contribution for this specific input.
- Positive impact (red) increases risk.
- Negative impact (green) decreases risk.

### 8) Raw inference data shown
- Expander section displays the exact feature vector sent to the model.

## Model details used

- **Model type:** `xgboost.sklearn.XGBClassifier`
- **Objective:** `binary:logistic`
- **n_estimators:** `200`
- **max_depth:** `4`
- **learning_rate:** `0.05`
- **subsample:** `1.0`
- **colsample_bytree:** `0.7`
- **random_state:** `42`

## Features used by the model

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

## Tech stack and what each tool does

- **Python**: core programming language.
- **Streamlit**: interactive web UI.
- **Pandas**: input table creation and column alignment.
- **NumPy**: numeric operations.
- **XGBoost**: risk prediction model.
- **scikit-learn**: model API compatibility utilities.
- **SHAP**: model explainability for local predictions.
- **Plotly**: interactive charts (gauge + feature impact bars).
- **Joblib**: loading saved model and feature artifacts.

## Installation and run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## How to use

1. Adjust applicant details in input panels.
2. Set the risk threshold in system controls.
3. Read:
   - Probability gauge
   - Final risk verdict
   - SHAP feature impact chart
4. Expand raw data section to inspect exact inference input.

## Notes

- Lower threshold = stricter policy (more applicants flagged risky).
- SHAP explains this specific prediction instance (local explanation).
- This is a decision-support tool and should be combined with policy/compliance checks.

## Author

- **Singhrituraj114**
