# 🛒 Customer Segmentation using Machine Learning
> 👥 Customer_Segmentation_System - This is an unsupervised ML model to cluster customers on the basis of their behavioral patterns.

## 📌 Overview
This project focuses on segmenting customers based on their demographics, spending behavior, and engagement patterns using unsupervised machine learning techniques. The goal is to enable targeted marketing, improve customer retention, and identify high-value customer groups.

---

## 🎯 Problem Statement
Businesses often struggle to understand diverse customer behavior. This project aims to:
- Identify distinct customer segments
- Analyze spending patterns
- Provide actionable business insights for each segment

---

## 📂 Dataset
- SmartCart Customer Dataset  
- Contains customer demographics, purchase behavior, and campaign responses  

---

## ⚠️ Note
- Model trained on key behavioral and demographic features for simplicity and interpretability.

---

## 📊 Evaluation

- Evaluated clustering performance using the **Elbow Method** to determine the optimal number of clusters. 
- Used **Silhouette Score** to measure cluster cohesion and separation.
- Combined quantitative evaluation with **cluster summary analysis** to ensure meaningful and interpretable customer segments.

---

## ⚙️ Project Workflow

```text
Raw Data
   ↓
Data Cleaning & Feature Engineering
   ↓
Encoding & Feature Scaling
   ↓
K-Means Clustering
   ↓
Cluster Analysis & Visualization
   ↓
Model Saving
   ↓
Streamlit Deployment 
