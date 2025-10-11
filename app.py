from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os
import logging

app = FastAPI()

model = joblib.load("trained_model.pkl")
scaler = joblib.load("trained_scaler.pkl")

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    age: int
    income: float
    gender: str

app = FastAPI(
    title="Ahadu SentriAI - Threat Detection API",
    description="AI-powered security model for anomaly detection and response",
    version="1.0.0"
)

# Global pipeline variable
pipeline = None

# Pydantic model with optional features
class Event(BaseModel):
    feature1: Optional[float] = 0
    feature2: Optional[float] = 0
    feature3: Optional[float] = 0


def convert_to_dataframe(incoming_data):
    """
    Convert incoming data to pandas DataFrame
    """
    if isinstance(incoming_data, dict):
        df = pd.DataFrame([incoming_data])
        return df
    elif isinstance(incoming_data, list):
        df = pd.DataFrame(incoming_data)
        return df
    else:
        raise ValueError("Data must be a dictionary or list of dictionaries")
# Define the input data model with all feature

# Lazy-load the pipeline
def load_pipeline():
    global pipeline
    if pipeline is None:
        model_path = "trained_model.pkl"
        scaler_path = "scaler.pkl"  # Rename from "scaler (1).pkl" for clarity
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logging.warning("Model or scaler not found, running in mock mode.")
            return False
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    return True

@app.on_event("startup")
def startup_event():
    load_assets()

@app.post("/predict")
def predict(data: InputData):
    if not load_assets():
        return {
    "error": "Model not found in CI environment",
    "anomaly_detected": False,
    "score": 0.0,
    "prediction": 0
}

    anomaly_score = abs(data.feature1) + abs(data.feature2) + abs(data.feature3)
    is_anomaly = anomaly_score > 5.0 or data.income < 1000 or data.age < 18

    input_array = np.array([[data.feature1, data.feature2, data.feature3, data.income]])
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)

    return {
        "anomaly_detected": is_anomaly,
        "score": anomaly_score,
        "prediction": int(prediction[0])
    }

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h1>EDR System is Live</h1>
    <p>Welcome to your FastAPI app!</p>
    <p>Try <a href='/dashboard'>/dashboard</a></p>
    """
