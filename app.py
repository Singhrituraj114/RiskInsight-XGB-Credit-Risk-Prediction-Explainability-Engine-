import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# ══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Risk Insight · Quantum",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. GLOBAL CSS (The "Eye-Catching" Magic)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Space+Mono:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif !important;
    background: #07091A !important;
    color: #E8EBF5 !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 0 2.5rem 4rem !important;
    max-width: 1400px !important;
}

/* Custom Sliders */
.stSlider > div > div > div:not([data-testid="stSliderTickBar"]) { background: rgba(0,212,255,0.15) !important; }
.stSlider > div > div > div > div:not([data-testid="stSliderThumbValue"]) { background: #00D4FF !important; box-shadow: 0 0 10px rgba(0,212,255,0.5); }
div[data-testid="stSliderThumbValue"] {
    background: transparent !important;
    box-shadow: none !important;
    color: #00D4FF !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 13px !important;
}
div[data-testid="stSliderTickBar"] { background: transparent !important; }
div[data-testid="stSliderTickBar"] p {
    color: #8892B0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    margin: 0 !important;
}
div[data-testid="stSlider"] label {
    color: #8892B0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Custom Number Inputs */
div[data-testid="stNumberInput"] label {
    color: #8892B0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
div[data-testid="stNumberInput"] input {
    background: rgba(0,212,255,0.05) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    border-radius: 8px !important;
    color: #00D4FF !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #00F5A0 !important;
    box-shadow: 0 0 0 2px rgba(0,245,160,0.2) !important;
    color: #00F5A0 !important;
}

/* Custom Expander */
.streamlit-expanderHeader {
    background: rgba(0,212,255,0.05) !important;
    border: 1px solid rgba(0,212,255,0.18) !important;
    border-radius: 12px !important;
    color: #00D4FF !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.06em;
}
.streamlit-expanderContent {
    background: rgba(7,9,26,0.9) !important;
    border: 1px solid rgba(0,212,255,0.1) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
}

/* Dataframe Styling */
.stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid rgba(0,212,255,0.2); }
.stDataFrame thead th {
    background: rgba(0,212,255,0.1) !important;
    color: #00D4FF !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. ML MODEL LOADING (Robust caching)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(ARTIFACTS_DIR / "model-RiskInsight-XGB.pkl")
        features = joblib.load(ARTIFACTS_DIR / "features-RiskInsight-XGB.pkl")
        return model, features
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, features = load_artifacts()

# ══════════════════════════════════════════════════════════════════════════════
# 4. HERO BANNER & ANIMATIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@keyframes pulse-dot {
  0%,100%{opacity:1;transform:scale(1);box-shadow:0 0 0 0 rgba(0,245,160,0.6)}
  50%{opacity:0.7;transform:scale(0.75);box-shadow:0 0 0 6px rgba(0,245,160,0)}
}
@keyframes shimmer {
  0%{background-position:200% center}
  100%{background-position:-200% center}
}
</style>

<div style="
    background: linear-gradient(135deg, #080D28 0%, #0D1B48 45%, #07122B 100%);
    border: 1px solid rgba(0,212,255,0.18);
    border-radius: 24px;
    padding: 3rem;
    margin: 1rem 0 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 30px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
">
  <div style="position:absolute;top:-100px;right:-50px;width:300px;height:300px;border-radius:50%;background:radial-gradient(circle,rgba(0,212,255,0.15) 0%,transparent 70%);pointer-events:none;"></div>
  <div style="position:absolute;bottom:-100px;left:5%;width:350px;height:350px;border-radius:50%;background:radial-gradient(circle,rgba(0,245,160,0.08) 0%,transparent 70%);pointer-events:none;"></div>

  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:2rem;position:relative;z-index:1;">
    <div>
      <div style="display:inline-flex;align-items:center;gap:9px;background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.2);border-radius:30px;padding:6px 16px;margin-bottom:1.5rem;">
        <div style="width:8px;height:8px;border-radius:50%;background:#00F5A0;animation:pulse-dot 2s ease-in-out infinite;"></div>
        <span style="font-family:'Space Mono',monospace;font-size:10px;color:#00D4FF;letter-spacing:0.12em;text-transform:uppercase;font-weight:700;">Engine Online · XGBoost Pipeline</span>
      </div>
      <h1 style="font-size:3.5rem;font-weight:900;color:#fff;line-height:1;margin:0 0 0.5rem;letter-spacing:-0.03em;">
        Credit Risk<br>
        <span style="background: linear-gradient(90deg, #00D4FF 0%, #00F5A0 50%, #00D4FF 100%); background-size: 200% auto; -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: shimmer 4s linear infinite;">Insight Quantum</span>
      </h1>
      <p style="font-size:16px;color:#8892B0;line-height:1.6;max-width:550px;margin:0;">
        Adjust the parameters below to evaluate financial risk in real-time. Powered by SHAP explainability for total transparency.
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5. INPUT PANELS (Cards Layout)
# ══════════════════════════════════════════════════════════════════════════════
col_inputs1, col_inputs2, col_threshold = st.columns([1.5, 1.5, 1], gap="large")

with col_inputs1:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(0,212,255,0.15); border-radius:20px; padding:1.5rem; box-shadow:0 4px 20px rgba(0,0,0,0.3); margin-bottom:0;">
      <h3 style="color:#00D4FF; font-size:1.2rem; margin-bottom:1rem; border-bottom:1px solid rgba(0,212,255,0.2); padding-bottom:0.5rem;">👤 Demographics & Financials</h3>
    """, unsafe_allow_html=True)
    
    age = st.slider("Applicant Age", 18, 100, 40)
    dep = st.slider("Number of Dependents", 0, 10, 0)
    cu = st.slider("Credit Utilization Ratio", 0.0, 1.0, 0.30, step=0.01)
    dr = st.slider("Debt-to-Income Ratio", 0.0, 5.0, 0.50, step=0.05)
    inc = st.number_input("Monthly Income ($)", min_value=0, max_value=1000000, value=50000, step=1000)
    st.markdown("</div>", unsafe_allow_html=True)

with col_inputs2:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,184,0,0.15); border-radius:20px; padding:1.5rem; box-shadow:0 4px 20px rgba(0,0,0,0.3); margin-bottom:0;">
      <h3 style="color:#FFB800; font-size:1.2rem; margin-bottom:1rem; border-bottom:1px solid rgba(255,184,0,0.2); padding-bottom:0.5rem;">📊 Credit & Payment History</h3>
    """, unsafe_allow_html=True)
    
    ocl = st.slider("Open Credit Lines", 0, 20, 5)
    rel = st.slider("Real Estate Loans", 0, 10, 1)
    l30 = st.slider("Late Payments (30-59 Days)", 0, 10, 0)
    l60 = st.slider("Late Payments (60-89 Days)", 0, 10, 0)
    l90 = st.slider("Late Payments (90+ Days)", 0, 10, 0)
    tpd = st.slider("Total Past Due Instances", 0, 20, 0)
    st.markdown("</div>", unsafe_allow_html=True)

with col_threshold:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,59,92,0.15); border-radius:20px; padding:1.5rem; box-shadow:0 4px 20px rgba(0,0,0,0.3); height: 100%;">
      <h3 style="color:#FF3B5C; font-size:1.2rem; margin-bottom:1rem; border-bottom:1px solid rgba(255,59,92,0.2); padding-bottom:0.5rem;">⚙️ System Controls</h3>
      <p style="color:#8892B0; font-size:0.9rem; line-height:1.4; margin-bottom: 1.5rem;">
        Adjust the strictness of the model. A lower threshold makes the engine more conservative (flags more users as risky).
      </p>
    """, unsafe_allow_html=True)
    
    threshold = st.slider("Risk Decision Threshold", 0.05, 0.95, 0.50, step=0.01)

    if threshold < 0.30:
        zone_label, zone_col = "STRICT POLICY", "#FF3B5C"
        zone_desc = "Conservative screening — more applicants get flagged as high risk."
    elif threshold < 0.60:
        zone_label, zone_col = "BALANCED POLICY", "#FFB800"
        zone_desc = "Standard sensitivity — typical industry risk tolerance."
    else:
        zone_label, zone_col = "LENIENT POLICY", "#00F5A0"
        zone_desc = "Relaxed screening — only clear high-risk cases get flagged."

    st.markdown(f"""
      <div style="margin-top:1.5rem; padding:1rem; background:{zone_col}14; border:1px solid {zone_col}44; border-radius:14px;">
        <div style="font-family:'Space Mono',monospace; font-size:10px; letter-spacing:0.1em; color:{zone_col}; font-weight:700; margin-bottom:6px;">● {zone_label}</div>
        <div style="font-size:12.5px; color:#8892B0; line-height:1.5;">{zone_desc}</div>
      </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 6. ML PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
input_data = pd.DataFrame([{
    "Age": age, "Dependents": dep, "CreditUtilization": cu, "DebtRatio": dr,
    "MonthlyIncome": inc, "OpenCreditLines": ocl, "RealEstateLoans": rel,
    "Late30_59": l30, "Late60_89": l60, "Late90Plus": l90, "TotalPastDue": tpd
}])

# Align columns perfectly with model training
input_data = input_data[features]

# Predict
prob = model.predict_proba(input_data)[0][1]
pred = int(prob > threshold)

# Dynamic Styling based on Threshold and Probability
if prob > threshold:
    verdict_txt = "HIGH RISK DETECTED"
    verdict_col = "#FF3B5C" # Neon Red
    verdict_bg  = "rgba(255,59,92,0.15)"
else:
    verdict_txt = "LOW RISK · APPROVED"
    verdict_col = "#00F5A0" # Neon Green
    verdict_bg  = "rgba(0,245,160,0.15)"

# ══════════════════════════════════════════════════════════════════════════════
# 7. RESULTS DASHBOARD (Gauge & Verdict)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
res_left, res_right = st.columns([1, 1.8], gap="large")

with res_left:
    st.markdown(f"""
    <div style="background:linear-gradient(145deg,rgba(255,255,255,0.03),rgba(0,0,0,0.3)); border:1px solid {verdict_col}44; border-radius:24px; padding:2rem; box-shadow:0 8px 30px rgba(0,0,0,0.5), 0 0 40px {verdict_col}22; text-align:center; height:100%;">
      <h3 style="color:#8892B0; font-family:'Space Mono', monospace; font-size:12px; text-transform:uppercase; letter-spacing:0.15em; margin-bottom: 0;">Probability Engine</h3>
    """, unsafe_allow_html=True)

    # Advanced Dark Mode Plotly Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        delta={
            'reference': threshold * 100,
            'increasing': {'color': "#FF3B5C"},
            'decreasing': {'color': "#00F5A0"},
            'font': {'size': 16, 'family': "Space Mono, monospace"},
        },
        number={"suffix": "%", "font": {"size": 60, "color": verdict_col, "family": "Outfit"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickmode": "array",
                "tickvals": [0, 20, 40, 60, 80, 100],
                "tickwidth": 1,
                "tickcolor": "rgba(255,255,255,0.2)",
                "tickfont": {"color": "#8892B0", "family": "Space Mono, monospace", "size": 11},
            },
            "bar": {"color": verdict_col, "thickness": 0.25},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, threshold * 100], "color": "rgba(0,245,160,0.10)"},
                {"range": [threshold * 100, 100], "color": "rgba(255,59,92,0.10)"},
            ],
            "threshold": {"line": {"color": "#FFB800", "width": 4}, "thickness": 0.75, "value": threshold * 100},
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=30, b=10, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", font={"family": "Outfit"})
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    # Verdict Badge
    st.markdown(f"""
      <div style="display:inline-block; padding:12px 32px; background:{verdict_bg}; border:1px solid {verdict_col}; border-radius:50px; box-shadow:0 0 20px {verdict_col}44; margin-top: -10px;">
        <span style="font-family:'Space Mono',monospace; font-size:14px; font-weight:700; color:{verdict_col}; letter-spacing:0.1em;">{verdict_txt}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 8. SHAP EXPLAINABILITY (The "Brain" of the model)
# ══════════════════════════════════════════════════════════════════════════════
with res_right:
    st.markdown("""
    <div style="background:linear-gradient(145deg,rgba(255,255,255,0.03),rgba(0,0,0,0.3)); border:1px solid rgba(0,212,255,0.15); border-radius:24px; padding:2rem; box-shadow:0 8px 30px rgba(0,0,0,0.5); height:100%;">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.75rem; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:1rem; margin-bottom:1rem;">
          <h3 style="color:#00D4FF; font-size:1.2rem; margin:0;">🧠 SHAP AI Explanation</h3>
          <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
            <span style="font-family:'Space Mono',monospace; font-size:9px; color:#FF3B5C; background:rgba(255,59,92,0.12); border:1px solid rgba(255,59,92,0.3); padding:3px 9px; border-radius:10px; white-space:nowrap;">▲ INCREASES RISK</span>
            <span style="font-family:'Space Mono',monospace; font-size:9px; color:#00F5A0; background:rgba(0,245,160,0.12); border:1px solid rgba(0,245,160,0.3); padding:3px 9px; border-radius:10px; white-space:nowrap;">▼ DECREASES RISK</span>
            <span style="font-family:'Space Mono',monospace; font-size:10px; color:#8892B0; letter-spacing:0.1em; background:rgba(255,255,255,0.05); padding:4px 10px; border-radius:12px;">LOCAL IMPACT</span>
          </div>
      </div>
    """, unsafe_allow_html=True)

    # Calculate SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)
    
    # Extract values safely
    base_val = shap_values.base_values[0]
    vals = shap_values.values[0]
    
    # Format for Plotly Waterfall/Bar
    shap_df = pd.DataFrame({"Feature": features, "Impact": vals})
    shap_df['Absolute Impact'] = shap_df['Impact'].abs()
    shap_df = shap_df.sort_values(by="Absolute Impact", ascending=True).tail(8) # Top 8 features
    
    # Color coding: Red pushes risk UP, Green pushes risk DOWN.
    # Opacity scales with magnitude so bigger contributors read as more intense.
    max_abs = shap_df['Absolute Impact'].max()
    shap_df['Color'] = [
        f"rgba(255,59,92,{0.45 + 0.55 * (abs_impact / max_abs):.3f})" if impact > 0
        else f"rgba(0,245,160,{0.45 + 0.55 * (abs_impact / max_abs):.3f})"
        for impact, abs_impact in zip(shap_df['Impact'], shap_df['Absolute Impact'])
    ]
    shap_df['Label'] = shap_df['Impact'].map(lambda v: f"{v:+.2f}")

    fig_shap = px.bar(
        shap_df, x="Impact", y="Feature", orientation='h',
        color="Color", color_discrete_map="identity",
        text="Label",
    )
    fig_shap.update_traces(
        textposition="outside",
        textfont=dict(color="#8892B0", family="Space Mono, monospace", size=11),
        marker_line_width=0,
        cliponaxis=False,
    )
    fig_shap.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(
                text="Impact on Risk Score (Log Odds)",
                font=dict(color="#8892B0", size=11)
            ),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="#8892B0"
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color="#E8EBF5", size=13)
        ),
        showlegend=False
    )
    st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})
    
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 9. FOOTER & RAW DATA EXPANDER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

with st.expander("🔍 VIEW RAW INFERENCE DATA & VECTORS"):
    st.dataframe(input_data, use_container_width=True, hide_index=True)

st.markdown(f"""
<div style="text-align:center; padding:3rem 0 1rem; font-family:'Space Mono',monospace; font-size:11px; color:rgba(136,146,176,0.5); letter-spacing:0.1em; text-transform:uppercase;">
  Built with <strong>XGBoost</strong> • <strong>Streamlit</strong> • <strong>SHAP</strong> <br>
  System Threshold: <span style="color:#FFB800;">{threshold}</span> &nbsp;|&nbsp; Computed Prob: <span style="color:{verdict_col};">{prob:.4f}</span>
</div>
""", unsafe_allow_html=True)
