import streamlit as st
import numpy as np
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_object

model = load_object("models/kmeans_model.pkl")
scaler = load_object("models/scaler.pkl")

st.title("🛒 Customer Segmentation System",text_alignment="center")
st.header("🔮 Predict customer segments using machine learning 🤖")

features = load_object("models/all_features.pkl")
income = st.number_input("Income")
spending = st.number_input("Total Spending")
age = st.number_input("Age")
children = st.number_input("Total Children")

input_dict = {
    "Income": income,
    "TotalSpending": spending,
    "Age": age,
    "TotalChildren": children
}
# Create dataframe with all columns
input_df = pd.DataFrame(columns=features)

# Fill known values
for col in input_dict:
    if col in input_df.columns:
        input_df[col] = [input_dict[col]]

# Handle missing columns 
input_df = input_df.fillna(0)