import streamlit as st
import numpy as np
import sys
import os
import pandas as pd

# import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_object

# Load model 
model = load_object("models/kmeans_model.pkl")
scaler = load_object("models/scaler.pkl")
features = load_object("models/all_features.pkl")

# ================================
# 🎯 UI
# ================================
st.title("🛒 Customer Segmentation System")
st.markdown("### 🎯 Get actionable customer insights instantly")

st.sidebar.header("Enter Customer Details")

income = st.sidebar.number_input("Income", min_value=0.0)
spending = st.sidebar.number_input("Total Spending", min_value=0.0)
age = st.sidebar.number_input("Age", min_value=0, step=1)
children = st.sidebar.number_input("Total Children", min_value=0, step=1)

# ================================
# 🧠 CLUSTER DEFINITIONS 
# ================================
cluster_info = {
    0: {
        "Name": "💰 Budget Individuals",
        "Insight": "Low income, low spending, mostly living alone",
        "Recommendation": "Offer discounts, coupons, and entry-level products to increase engagement."
    },
    1: {
        "Name": "👨‍👩‍👧 Family Customers",
        "Insight": "Low income but partnered households with moderate engagement",
        "Recommendation": "Provide bundle offers and family-oriented promotions."
    },
    2: {
        "Name": "🛍️ Active Shoppers",
        "Insight": "Moderate income with high purchasing activity and frequent online shopping, mostly living with partner",
        "Recommendation": "Target with personalized recommendations and loyalty rewards."
    },
    3: {
        "Name": "💎 High Value Customers",
        "Insight": "High income customers with high spending, mostly living with partners",
        "Recommendation": "Offer premium services, trending exclusive deals, and VIP loyalty programs."
    }
}

# ================================
# 🔮 PREDICTION
# ================================
if st.button("Predict Customer Segment"):

    # full feature dataframe
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

    # Handle missing columns
    input_df = input_df.fillna(0)

    # Scale input
    data_scaled = scaler.transform(input_df)

    # Predict cluster
    cluster = model.predict(data_scaled)[0]

    # Get cluster info
    info = cluster_info.get(cluster)

    # ================================
    # 📊 OUTPUT
    # ================================
    st.subheader("📊 Prediction Result")

    st.markdown(f"### Segment: {info['Name']}")
    st.write(f"**Insight:** {info['Insight']}")
    st.write(f"**Recommendation:** {info['Recommendation']}")