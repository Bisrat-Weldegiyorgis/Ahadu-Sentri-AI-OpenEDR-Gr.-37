# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(
    title="Ahadu SentriAI - Threat Detection API",
    description="AI-powered security model for anomaly detection and response",
    version="1.0.0"
)

# Global placeholder for the pipeline
pipeline = None

def load_pipeline():
    """
    Lazy-load the model pipeline on first use.
    """
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

# Define input data model
class Event(BaseModel):
    feature1: float
    feature2: float
    feature3: float

def extract_features(event: Event):
    """
    Convert Event into feature list for prediction
    """
    return [event.feature1, event.feature2, event.feature3]

@app.post("/ingest")
async def ingest(event: Event):
    """
    Endpoint to receive event and return decision
    """
    model = load_pipeline()
    features = extract_features(event)

    try:
        # Make prediction using pipeline
        pred = model.predict([features])[0]
        decision = "approve" if pred == 1 else "reject"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"decision": decision}
