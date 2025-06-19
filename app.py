# src/app.py

import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path

# Set root directory based on this file's location
BASE_DIR = Path(__file__).resolve().parent.parent

# Build absolute path to model and scaler
model_path = BASE_DIR / 'outputs' / 'metrics' / 'LogisticRegression_model.pkl'
scaler_path = BASE_DIR / 'outputs' / 'metrics' / 'scaler.pkl'

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load the saved model and scaler
# model = joblib.load('../outputs/metrics/LogisticRegression_model.pkl')
# scaler = joblib.load('../outputs/metrics/scaler.pkl')

st.write("Enter your health metrics below:")

# 🔧 Add all 8 features used during training
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.slider("Age", min_value=21, max_value=100, value=33)

if st.button("Predict"):
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                               insulin, bmi, dpf, age]],
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ High risk of Diabetes")
    else:
        st.success("✅ Low risk of Diabetes")