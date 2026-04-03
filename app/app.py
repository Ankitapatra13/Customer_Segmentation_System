import streamlit as st
import numpy as np
import sys
import os
import pandas as pd

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_object

# Load model and scaler
model = load_object("models/kmeans_model.pkl")
scaler = load_object("models/scaler.pkl")
features = load_object("models/all_features.pkl")

# UI
st.title("🛒 Customer Segmentation System")
st.subheader("🔮 Predict customer segments using machine learning")

# Inputs
income = st.number_input("Income", min_value=0.0)
spending = st.number_input("Total Spending", min_value=0.0)
age = st.number_input("Age", min_value=0.0)
children = st.number_input("Total Children", min_value=0.0)

if st.button("Predict"):

    # Create full feature dataframe
    input_df = pd.DataFrame(columns=features)

    input_dict = {
        "Income": income,
        "TotalSpending": spending,
        "Age": age,
        "TotalChildren": children
    }

    # Fill known values
    for col in input_dict:
        if col in input_df.columns:
            input_df.loc[0, col] = input_dict[col]

    # Fill remaining columns
    input_df = input_df.fillna(0)

    # Scale FULL dataframe
    data_scaled = scaler.transform(input_df)

    # Predict
    cluster = model.predict(data_scaled)[0]

    st.success(f"Customer belongs to Cluster {cluster}")