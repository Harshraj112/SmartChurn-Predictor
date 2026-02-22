import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import base64
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartChurn Predictor",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Load Model & Artifacts ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

@st.cache_resource
def load_encoders():
    with open('label_encoder_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        ohe_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return le_gender, ohe_geo, scaler

model = load_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_encoders()

# ─── Helper: image → base64 ────────────────────────────────────────────────────
def img_to_b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

hero_b64    = img_to_b64("assets/hero_banner.png")
warning_b64 = img_to_b64("assets/churn_warning.png")

# ─── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.main { background: #ffffff !important; }
section[data-testid="stSidebar"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero section ── */
.hero-wrap {
    display: flex;
    align-items: center;
    gap: 2.5rem;
    background: #ffffff;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid #eef0f4;
    box-shadow: 0 4px 24px rgba(99,102,241,.06);
}
.hero-text h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: #1e1b4b;
    margin: 0 0 .6rem 0;
    line-height: 1.2;
}
.hero-text p {
    color: #6b7280;
    font-size: 1.05rem;
    line-height: 1.7;
    margin: 0;
    max-width: 520px;
}
.hero-badge {
    display: inline-block;
    background: #eef2ff;
    color: #4f46e5;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    padding: .3rem .8rem;
    border-radius: 999px;
    margin-bottom: .9rem;
}
.hero-img {
    flex-shrink: 0;
    width: 260px;
    border-radius: 16px;
    overflow: hidden;
}
.hero-img img { width: 100%; display: block; }

/* ── Section heading ── */
.section-label {
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 1rem;
}

/* ── Card wrappers ── */
.card {
    background: #ffffff;
    border: 1px solid #eef0f4;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 2px 12px rgba(99,102,241,.05);
    margin-bottom: 1.4rem;
}
.card-title {
    font-size: .85rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}

/* ── Streamlit widget tweaks ── */
.stSlider > div > div > div { background: #4f46e5 !important; }
.stSelectbox > div > div { border-radius: 10px !important; border-color: #e5e7eb !important; }
.stNumberInput > div > div { border-radius: 10px !important; border-color: #e5e7eb !important; }
label { font-size: .85rem !important; font-weight: 500 !important; color: #374151 !important; }

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: .8rem 2.4rem !important;
    border-radius: 12px !important;
    border: none !important;
    width: 100% !important;
    letter-spacing: .02em !important;
    transition: opacity .2s ease !important;
    box-shadow: 0 4px 14px rgba(79,70,229,.35) !important;
}
.stButton > button:hover { opacity: .88 !important; }

/* ── Result cards ── */
.result-churn {
    background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
    border: 1.5px solid #fca5a5;
    border-radius: 20px;
    padding: 2rem 2.4rem;
    display: flex;
    align-items: center;
    gap: 2rem;
}
.result-safe {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1.5px solid #86efac;
    border-radius: 20px;
    padding: 2rem 2.4rem;
    display: flex;
    align-items: center;
    gap: 2rem;
}
.result-img { width: 130px; flex-shrink: 0; border-radius: 12px; overflow: hidden; }
.result-img img { width: 100%; display: block; }
.result-body h2 { margin: 0 0 .4rem 0; font-size: 1.65rem; font-weight: 700; }
.result-body p  { margin: 0; color: #6b7280; font-size: .95rem; }
.result-churn .result-body h2 { color: #dc2626; }
.result-safe  .result-body h2 { color: #16a34a; }

/* ── Probability bar ── */
.prob-wrap { margin-top: 1.4rem; }
.prob-label {
    font-size: .82rem;
    font-weight: 600;
    color: #6b7280;
    margin-bottom: .4rem;
    display: flex;
    justify-content: space-between;
}
.prob-track {
    height: 10px;
    background: #f3f4f6;
    border-radius: 999px;
    overflow: hidden;
}
.prob-fill-red {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #f87171, #dc2626);
    transition: width .8s ease;
}
.prob-fill-green {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #4ade80, #16a34a);
    transition: width .8s ease;
}

/* ── Stat pills ── */
.pills-row { display: flex; gap: .8rem; flex-wrap: wrap; margin-top: 1.2rem; }
.pill {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 999px;
    padding: .35rem 1rem;
    font-size: .78rem;
    font-weight: 500;
    color: #374151;
}
.pill span { color: #4f46e5; font-weight: 700; }

/* ── Divider ── */
.custom-divider {
    height: 1px; background: #f3f4f6; margin: 1.6rem 0;
}

/* ── Footer ── */
.custom-footer {
    text-align: center; color: #9ca3af; font-size: .8rem;
    margin-top: 3rem; padding-top: 1.5rem;
    border-top: 1px solid #f3f4f6;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero Section ──────────────────────────────────────────────────────────────
hero_img_tag = f'<img src="data:image/png;base64,{hero_b64}" />' if hero_b64 else ""
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-text">
        <div class="hero-badge">AI-Powered Analytics</div>
        <h1>SmartChurn<br>Predictor</h1>
        <p>Enter a customer's profile details below and our deep learning model
        will instantly calculate their churn probability — helping your team
        act before it's too late.</p>
    </div>
    {"<div class='hero-img'>" + hero_img_tag + "</div>" if hero_img_tag else ""}
</div>
""", unsafe_allow_html=True)

# ─── Input Form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card"><div class="card-title">🌍 Demographics</div>', unsafe_allow_html=True)
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender    = st.selectbox('Gender', label_encoder_gender.classes_)
    age       = st.slider('Age', 18, 92, 35)
    tenure    = st.slider('Tenure (years with bank)', 0, 10, 3)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">💳 Product Usage</div>', unsafe_allow_html=True)
    num_of_products  = st.slider('Number of Products', 1, 4, 1)
    has_cr_card      = st.selectbox('Has Credit Card', ['No', 'Yes'])
    is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'])
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card"><div class="card-title">💰 Financials</div>', unsafe_allow_html=True)
    credit_score     = st.number_input('Credit Score', min_value=300, max_value=900, value=650)
    balance          = st.number_input('Account Balance ($)', min_value=0.0, value=60000.0, step=500.0)
    estimated_salary = st.number_input('Estimated Annual Salary ($)', min_value=0.0, value=50000.0, step=1000.0)
    st.markdown('</div>', unsafe_allow_html=True)

    # Summary pills
    st.markdown(f"""
    <div class="card" style="margin-top:0">
        <div class="card-title">📋 Quick Summary</div>
        <div class="pills-row">
            <div class="pill">📍 <span>{geography}</span></div>
            <div class="pill">👤 <span>{gender}</span></div>
            <div class="pill">🎂 Age <span>{age}</span></div>
            <div class="pill">📅 Tenure <span>{tenure}y</span></div>
            <div class="pill">📦 Products <span>{num_of_products}</span></div>
            <div class="pill">💳 Card <span>{'Yes' if has_cr_card=='Yes' else 'No'}</span></div>
            <div class="pill">⚡ Active <span>{'Yes' if is_active_member=='Yes' else 'No'}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Predict Button ────────────────────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("🔍  Run Churn Analysis")

# ─── Prediction Logic ──────────────────────────────────────────────────────────
if predict_clicked:
    has_cr_card_val      = 1 if has_cr_card == 'Yes' else 0
    is_active_member_val = 1 if is_active_member == 'Yes' else 0

    input_data = pd.DataFrame({
        'CreditScore':     [credit_score],
        'Gender':          [label_encoder_gender.transform([gender])[0]],
        'Age':             [age],
        'Tenure':          [tenure],
        'Balance':         [balance],
        'NumOfProducts':   [num_of_products],
        'HasCrCard':       [has_cr_card_val],
        'IsActiveMember':  [is_active_member_val],
        'EstimatedSalary': [estimated_salary],
    })

    geo_encoded    = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_data     = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_scaled   = scaler.transform(input_data)

    prediction      = model.predict(input_scaled)
    churn_prob      = float(prediction[0][0])
    churn_pct       = round(churn_prob * 100, 1)
    will_churn      = churn_prob > 0.5

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)

    if will_churn:
        warn_img_tag = f'<img src="data:image/png;base64,{warning_b64}" />' if warning_b64 else "⚠️"
        st.markdown(f"""
        <div class="result-churn">
            <div class="result-img">{warn_img_tag}</div>
            <div class="result-body">
                <h2>High Churn Risk</h2>
                <p>This customer has a <strong>{churn_pct}%</strong> probability of leaving.
                   Proactive retention steps are recommended.</p>
                <div class="prob-wrap">
                    <div class="prob-label"><span>Churn Probability</span><span>{churn_pct}%</span></div>
                    <div class="prob-track">
                        <div class="prob-fill-red" style="width:{churn_pct}%;"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        retain_pct = round((1 - churn_prob) * 100, 1)
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-img" style="font-size:5rem;text-align:center;padding:1rem;">✅</div>
            <div class="result-body">
                <h2>Low Churn Risk</h2>
                <p>This customer has a <strong>{retain_pct}%</strong> probability of staying.
                   Keep delivering value to maintain loyalty.</p>
                <div class="prob-wrap">
                    <div class="prob-label"><span>Retention Probability</span><span>{retain_pct}%</span></div>
                    <div class="prob-track">
                        <div class="prob-fill-green" style="width:{retain_pct}%;"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Key factors note
    st.markdown(f"""
    <div class="card" style="margin-top:1.4rem">
        <div class="card-title">🧮 Model Input Snapshot</div>
        <div class="pills-row">
            <div class="pill">Score <span>{credit_score}</span></div>
            <div class="pill">Balance <span>${balance:,.0f}</span></div>
            <div class="pill">Salary <span>${estimated_salary:,.0f}</span></div>
            <div class="pill">Products <span>{num_of_products}</span></div>
            <div class="pill">Tenure <span>{tenure}y</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="custom-footer">
    Built with ❤️ by <strong>Harsh Raj</strong> &nbsp;·&nbsp; Powered by TensorFlow &amp; Streamlit
</div>
""", unsafe_allow_html=True)
