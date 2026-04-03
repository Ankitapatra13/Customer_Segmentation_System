import streamlit as st
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_object

model = load_object("models/kmeans_model.pkl")
scaler = load_object("models/scaler.pkl")

st.title("Customer Segmentation App")

income = st.number_input("Income")
spending = st.number_input("Total Spending")
age = st.number_input("Age")
children = st.number_input("Total Children")

if st.button("Predict"):
    data = np.array([[income, spending, age, children]])
    data_scaled = scaler.transform(data)

    cluster = model.predict(data_scaled)[0]

    st.success(f"Customer belongs to Cluster {cluster}")