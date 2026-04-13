# ================================
# 🚀 INSURANCE COST PREDICTION APP
# ================================

import streamlit as st
import pandas as pd
import joblib

# 🔥 FIX: PATCH sklearn internal error (_RemainderColsList)
import sklearn.compose._column_transformer as ct

if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList

# ================================
# IMPORTANT IMPORTS (REQUIRED)
# ================================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Insurance Predictor",
    page_icon="💰",
    layout="wide"
)

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except Exception as e:
        st.error(f"❌ Model Loading Failed: {e}")
        return None

# ================================
# UI DESIGN
# ================================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
h1 {
    text-align: center;
    color: #00ffe1;
}
.stButton>button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    background: linear-gradient(45deg, #00ffe1, #007cf0);
}
.result-box {
    background: linear-gradient(45deg, #00ffe1, #00c6ff);
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 26px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>💰 Insurance Cost Prediction</h1>", unsafe_allow_html=True)

# ================================
# LOAD MODEL
# ================================
model = load_model()

if model is None:
    st.stop()

# ================================
# INPUTS
# ================================
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 25)
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    children = st.slider("Children", 0, 5, 0)

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# ================================
# PREDICTION
# ================================
if st.button("🚀 Predict Insurance Cost"):

    try:
        input_df = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region]
        })

        with st.spinner("🔮 Predicting..."):
            prediction = model.predict(input_df)[0]

        st.markdown(
            f'<div class="result-box">💰 Estimated Cost: ₹ {prediction:,.2f}</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
