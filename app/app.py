import streamlit as st
import numpy as np
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
        "Name": "👨‍👩‍👧 Family Budget Customers",
        "Insight": "Moderate income, low spending, larger families, older age group",
        "Recommendation": "Offer family bundles, discounts, and value-for-money deals"
    },
    1: {
        "Name": "🛍️ Affluent Senior Customers",
        "Insight": "High income, high spending, older customers with fewer dependents",
        "Recommendation": "Target with premium services, comfort products, and loyalty programs"
    },
    2: {
        "Name": "💰 Young Budget Customers",
        "Insight": "Low income, low spending, younger demographic with fewer children",
        "Recommendation": "Engage with discounts, entry-level products, and promotional offers"
    },
    3: {
        "Name": "💎 Premium High-Value Customers",
        "Insight": "Very high income and extremely high spending with minimal family burden",
        "Recommendation": "Provide VIP experiences, exclusive deals, and personalized services"
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

    # Fill values
    for col in input_dict:
        if col in input_df.columns:
            input_df.loc[0, col] = input_dict[col]

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
    st.subheader("📍Position in market")
    st.write(f"**Income:** {income}")
    st.write(f"**Spending:** {spending}")

    # Load dataset for visualization
    df = pd.read_csv("models/clustered_data.csv")
    df = df[["Income", "TotalSpending","Cluster"]]

    fig, ax = plt.subplots()
    ax.scatter(df["Income"], df["TotalSpending"], c=df["Cluster"], cmap="tab10", alpha=0.6)
    ax.scatter(income, spending, marker="X", s=200 , color="red")  # curr customer point
    ax.set_xlabel("Income")
    ax.set_ylabel("Total Spending")
    ax.set_title("Customer Distribution")
    st.pyplot(fig) 

    ### adding cluster summary
    summary = pd.read_csv("models/cluster_summary.csv")
    st.subheader("📊 Cluster Overview")
    st.dataframe(summary)
