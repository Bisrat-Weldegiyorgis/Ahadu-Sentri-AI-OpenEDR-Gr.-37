from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

model = None
scaler = None

def load_assets():
    global model, scaler
    if model is None or scaler is None:
        if not os.path.exists("trained_model.pkl") or not os.path.exists("scaler (1).pkl"):
            print("[Warning] Model or scaler not found, running in mock mode.")
            return False
        model = joblib.load("trained_model.pkl")
        scaler = joblib.load("scaler (1).pkl")
    return True

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    age: int
    income: float
    gender: str




@app.post("/predict")
async def predict(data: InputData):
    input_array = np.array([[data.feature1, data.feature2, data.feature3]])

    if model is None or scaler is None:
        print("Warning: Model or scaler not found, running in mock mode.")
        # Simple mock logic: flag as anomaly if any feature is unusually high
        threshold = 1000
        is_anomalous = any(x > threshold for x in input_array[0])
        return {"anomaly_detected": is_anomalous}

    # prediction logic
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    is_anomalous = bool(prediction[0])
    return {"anomaly_detected": is_anomalous}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
    <html>
        <head><title>Dashboard</title></head>
        <body>
            <h1>EDR Dashboard</h1>
            <p>Model status: {}</p>
            <p>Scaler status: {}</p>
        </body>
    </html>
    """.format("Loaded" if model else "Missing", "Loaded" if scaler else "Missing")



@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h1>EDR System is Live</h1>
    <p>Welcome to your FastAPI app!</p>
    <p>Try <a href='/dashboard'>/dashboard</a></p>
    """

