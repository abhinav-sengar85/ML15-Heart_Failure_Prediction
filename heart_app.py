import streamlit as st
import pandas as pd
import numpy as np
import joblib


# load the trained model and scaler
heart_model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Heart Disease Prediction App by Abhinav")
st.image("https://my.clevelandclinic.org/-/scassets/images/org/health/articles/21704-heart-overview-outside", caption="Heart Disease", width=200)

age = st.slider("Age", min_value=1, max_value=120, value=30)
anaemia = st.selectbox("Anemia (0 = No, 1 = Yes)", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0, max_value=10000, value=100)
diabities = st.selectbox("Diabetes (0 = No, 1 = Yes)", [0, 1])
ejection_fraction = st.slider("Ejection Fraction (%)", min_value=0, max_value=100, value=50)
hbp = st.selectbox("High Blood Pressure (0 = No, 1 = Yes)", [0, 1])
platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0, max_value=1000000, value=250000)
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=20.0, value=1.0)
gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
smoking = st.selectbox("Smoking (0 = No, 1 = Yes)", [0, 1])


if st.button("Heart Failure Prediction"):
    input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabities, ejection_fraction, hbp, platelets, serum_creatinine, gender, smoking]])
    input_data_scaled = scaler.transform(input_data)
    prediction = heart_model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.error("The patient is likely to have heart failure.")
    else:
        st.success("The patient is unlikely to have heart failure.")