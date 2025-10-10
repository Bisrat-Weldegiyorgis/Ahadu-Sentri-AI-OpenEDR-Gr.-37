from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import os
import numpy as np

app = FastAPI(
    title="Ahadu SentriAI Automated EDR Security System",
    description="""AI-powered security model for anomaly detection and response:
Detects anomalies in real-time and triggers automated responses based on threat level.""",
    version="1.0"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Automated EDR Security System"}

    

from pydantic import BaseModel

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature2: float
   

# load

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")  


# Extract features from the Pydantic model
def extract_features(event: Event):
    return np.array([[event.feature1, event.feature2, event.feature3]])

# Make decision based on prediction
def decide(prediction):
    # Example logic
    return "approve" if prediction[0] == 1 else "reject"

# Ingest endpoint
@app.post("/ingest")
async def ingest(event: Event):
    model = load_pipeline()
    features = extract_features(event)
    prediction = model.predict(features)
    decision = decide(prediction)
    return {"decision": decision, "prediction": list(prediction)}

