import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("weekly_sales_model.h5")

# Set page configuration
st.set_page_config(page_title="🛒 Weekly Sales Predictor", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            font-size:40px !important;
            color: #4CAF50;
            text-align: center;
            font-weight: bold;
        }
        .subtitle {
            font-size:20px !important;
            color: #666;
            text-align: center;
            margin-bottom: 30px;
        }
        .footer {
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            color: #999;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="title">🧠 Weekly Sales Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict weekly sales based on fuel price, temperature, CPI, unemployment & holiday flag</div>', unsafe_allow_html=True)

# Create input fields
col1, col2 = st.columns(2)

with col1:
    fuel_price = st.number_input("⛽ Fuel Price ($)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
    temperature = st.number_input("🌡️ Temperature (°F)", min_value=-30.0, max_value=130.0, value=70.0, step=1.0)
    holiday_flag = st.selectbox("🏖️ Holiday Week?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    cpi = st.number_input("📊 Consumer Price Index (CPI)", min_value=100.0, max_value=300.0, value=220.0, step=0.1)
    unemployment = st.number_input("📉 Unemployment Rate (%)", min_value=0.0, max_value=20.0, value=7.5, step=0.1)

# Predict button
if st.button("🔍 Predict Weekly Sales"):
    input_data = np.array([[fuel_price, temperature, holiday_flag, unemployment, cpi]])
    prediction = model.predict(input_data)[0][0]
    formatted_prediction = f"${prediction:,.2f}"

    st.success(f"🧾 **Predicted Weekly Sales:** {formatted_prediction}")

# Footer
st.markdown('<div class="footer">Made with ❤️ using Streamlit & TensorFlow</div>', unsafe_allow_html=True)
