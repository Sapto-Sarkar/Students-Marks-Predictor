import pandas as pd
import joblib
from fastapi import FastAPI

app = FastAPI()

loaded_model = joblib.load("students_marks.pkl")

@app.get("/")
def home():
    return {"API is running well"}

@app.post("/predict")
def predict(hours_studied: float, sleep_hours: float, attendance_percent: float, previous_scores: float):
    data = pd.DataFrame(
        [[hours_studied, sleep_hours, attendance_percent, previous_scores]],
        columns=["hours_studied", "sleep_hours", "attendance_percent", "previous_scores"]
    )
    prediction = loaded_model.predict(data)[0]
    return {"Your next exam marks would be": round(float(prediction),2)}
