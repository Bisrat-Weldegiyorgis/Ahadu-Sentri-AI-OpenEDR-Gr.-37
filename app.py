from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import os
import numpy as np

app = FastAPI(
    title="Ahadu SentriAI - Threat Detection API",
    description="AI-powered security model for anomaly detection and response: 
       The Threat Detection API is a robust, AI-powered endpoint designed to analyze incoming data and identify potential security threats in real time. 
       Built with FastAPI and integrated with a trained machine learning model, it enables automated decision-making for cybersecurity workflows, intrusion detection systems, and enterprise-grade monitoring tools.",
    version="1.0.0"
)

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

