from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model and scaler
model = joblib.load("trained_model.pkl")
scaler = joblib.load("scaler (1).pkl")

# Input schema
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    age: int
    income: float
    gender: str

@app.post("/predict")
def predict(data: InputData):
    # Anomaly detection logic
    anomaly_score = abs(data.feature1) + abs(data.feature2) + abs(data.feature3)
    is_anomaly = anomaly_score > 5.0 or data.income < 1000 or data.age < 18

    # Model prediction (using only feature1 and feature2)
    input_array = np.array([[data.feature1, data.feature2, data.feature3, data.income]])
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)

    return {
        "anomaly_detected": is_anomaly,
        "score": anomaly_score,
        "prediction": int(prediction[0])
    }

@app.get("/")
def read_root():
    return {"message": "EDR system is live"}
