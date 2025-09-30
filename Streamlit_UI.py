import warnings
import joblib
import streamlit as st
import numpy as np

# Load model with joblib
model = joblib.load("students_marks.pkl")

st.title("Predict Your Marks")

f1 = st.number_input("Hours Studied Daily")
f2 = st.number_input("Sleep Hours Daily")
f3 = st.number_input("Attendance Percent")
f4 = st.number_input("Previous Exam Scores")

if st.button("Predict"):
    pred = model.predict([[f1, f2, f3, f4]])
    st.success(f"Your next exam score will be: {round(pred[0],2)}")
