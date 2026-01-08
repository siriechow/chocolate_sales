# ============================================================
# Streamlit App: Boxes Shipped Classification
# ============================================================
"""
Objective:
Predict shipment category (Low / Medium / High) based on
sales inputs using the best trained ML model.

Model:
- best_model.pkl
"""

# ============================================================
# Library Imports
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Shipment Prediction App",
    page_icon="📦",
    layout="wide"
)

# ============================================================
# Load Model
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# ============================================================
# App Title & Description
# ============================================================
st.title("📦 Shipment Volume Classification App")
st.markdown("""
### Predict **Boxes Shipped Category** using Machine Learning  

This app classifies shipment volume into:
- **Low**
- **Medium**
- **High**

Built for **competition-grade evaluation**, focusing on clarity,
business relevance, and interactive user experience.
""")

# ============================================================
# Sidebar Inputs
# ============================================================
st.sidebar.header("🔧 Input Parameters")

sales_person = st.sidebar.number_input(
    "Sales Person (Encoded)",
    min_value=0,
    max_value=100,
    value=10
)

country = st.sidebar.number_input(
    "Country (Encoded)",
    min_value=0,
    max_value=10,
    value=2
)

product = st.sidebar.number_input(
    "Product (Encoded)",
    min_value=0,
    max_value=50,
    value=5
)

amount = st.sidebar.number_input(
    "Sales Amount",
    min_value=0.0,
    max_value=100000.0,
    value=5000.0
)

year = st.sidebar.number_input("Year", 2015, 2030, 2024)
month = st.sidebar.slider("Month", 1, 12, 6)
quarter = st.
