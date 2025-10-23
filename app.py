import streamlit as st
import numpy as np
import joblib

model = joblib.load('model.pkl')

st.set_page_config(page_title="Customer Segmentation App", page_icon="👥")
st.title("👥 Customer Segmentation Using Machine Learning")
st.write("Enter customer details below to predict the segment they belong to.")

income = st.number_input("Annual Income (in thousands)", min_value=0.0, step=0.5)
score = st.number_input("Spending Score (1-100)", min_value=0.0, step=1.0)

if st.button("Predict Segment"):
    features = np.array([[income, score]])
    cluster = model.predict(features)[0]
    st.success(f"The customer belongs to Cluster {cluster}")

st.markdown("---")
st.caption("Developed by MVS Pragnya Bala")
