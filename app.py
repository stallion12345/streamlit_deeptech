import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0a;
    color: #f0f0f0;
}

.stApp {
    background-color: #0a0a0a;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
}

.fraud-badge {
    background: #ff2d2d;
    color: white;
    padding: 1.2rem 2rem;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 2px;
    border-left: 6px solid #ff0000;
}

.safe-badge {
    background: #0d3320;
    color: #00ff88;
    padding: 1.2rem 2rem;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 2px;
    border-left: 6px solid #00ff88;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 4px;
    color: #888;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background-color: #1a1a1a;
    color: #f0f0f0;
    border: 1px solid #333;
    border-radius: 4px;
}

.stButton > button {
    background-color: #f0f0f0;
    color: #0a0a0a;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 2px;
    border: none;
    border-radius: 4px;
    padding: 0.6rem 2rem;
    width: 100%;
    font-size: 0.85rem;
}

.stButton > button:hover {
    background-color: #00ff88;
    color: #0a0a0a;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 2px;
    color: #888;
}

.stTabs [aria-selected="true"] {
    color: #f0f0f0;
    border-bottom: 2px solid #00ff88;
}

.result-row {
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid #1a1a1a;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}

hr {
    border-color: #1a1a1a;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('fraud_model.pkl')

model = load_model()

# ── Feature columns (V1-V28 + Amount) ─────────────────────────────────────────
FEATURE_COLS = [f'V{i}' for i in range(1, 29)] + ['Amount']

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ Fraud Detection System")
st.markdown("<p class='section-header'>Powered by Logistic Regression + SMOTE</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["SINGLE TRANSACTION", "BATCH CSV UPLOAD"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single transaction input
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<p class='section-header'>Enter Transaction Details</p>", unsafe_allow_html=True)
    st.caption("V1–V28 are PCA-transformed features from the original dataset. Enter 0 if unknown.")

    cols = st.columns(4)
    input_values = {}

    for i, feature in enumerate(FEATURE_COLS):
        with cols[i % 4]:
            input_values[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.4f",
                key=f"input_{feature}"
            )

    st.markdown("")
    if st.button("ANALYSE TRANSACTION", key="single_predict"):
        input_df = pd.DataFrame([input_values])
        prediction = model.predict(input_df)[0]

        st.markdown("---")
        st.markdown("<p class='section-header'>Result</p>", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown("<div class='fraud-badge'>⚠️ FRAUDULENT TRANSACTION</div>", unsafe_allow_html=True)
            st.error("This transaction has been flagged as fraudulent. Recommend blocking or manual review.")
        else:
            st.markdown("<div class='safe-badge'>✓ LEGITIMATE TRANSACTION</div>", unsafe_allow_html=True)
            st.success("This transaction appears legitimate. No action required.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CSV batch upload
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<p class='section-header'>Upload CSV File</p>", unsafe_allow_html=True)
    st.caption("CSV must contain columns: V1 to V28 and Amount. Class column is optional.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)

        # Check required columns
        missing_cols = [c for c in FEATURE_COLS if c not in df_upload.columns]

        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            X_batch = df_upload[FEATURE_COLS]
            predictions = model.predict(X_batch)
            df_upload['Prediction'] = predictions
            df_upload['Result'] = df_upload['Prediction'].map({0: 'Legitimate', 1: 'Fraud'})

            # Summary
            total = len(df_upload)
            fraud_count = int((predictions == 1).sum())
            legit_count = int((predictions == 0).sum())

            st.markdown("---")
            st.markdown("<p class='section-header'>Batch Results Summary</p>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Transactions", total)
            c2.metric("Flagged as Fraud", fraud_count)
            c3.metric("Legitimate", legit_count)

            st.markdown("---")
            st.markdown("<p class='section-header'>Full Results</p>", unsafe_allow_html=True)
            st.dataframe(
                df_upload[FEATURE_COLS + ['Result']].style.applymap(
                    lambda v: 'color: #ff2d2d; font-weight: bold' if v == 'Fraud' else 'color: #00ff88',
                    subset=['Result']
                ),
                use_container_width=True
            )

            # Download
            csv_out = df_upload.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="DOWNLOAD RESULTS CSV",
                data=csv_out,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-family:Space Mono,monospace; font-size:0.7rem; color:#444;'>"
    "Credit Card Fraud Detection · Logistic Regression · SMOTE Balanced · ROC-AUC 0.960"
    "</p>",
    unsafe_allow_html=True
)
