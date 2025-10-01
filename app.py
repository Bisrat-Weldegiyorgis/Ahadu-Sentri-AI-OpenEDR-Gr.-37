from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import os
import numpy as np

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

# Lazy-load the pipeline
def load_pipeline():
    global pipeline
    if pipeline is None:
        model_path = "trained_model.pkl"
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file '{model_path}' not found.")
        try:
            pipeline = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    return pipeline

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

