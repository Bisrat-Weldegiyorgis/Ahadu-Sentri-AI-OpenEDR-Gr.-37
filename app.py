from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="Ahadu SentriAI - Threat Detection API",
    description="AI-powered security model for anomaly detection and response. "
                "A lightweight API for serving endpoint telemetry ML models (Ahadu SentriAI).",
    version="1.0.0"
)

# Load the trained model and scaler
pipeline = joblib.load("trained_model.pkl")

# Define input schema using Pydantic
class Event(BaseModel):
    feature1: float
    feature2: float
    feature3: float

# Utility function: extract features into numpy array
def extract_features(event: Event):
    return np.array([[event.feature1, event.feature2, event.feature3]])

# Decision logic
def decide(prediction, probability, features):
    if prediction == 1 and probability > 0.8:
        return "threat_detected"
    elif prediction == 1:
        return "suspicious"
    else:
        return "safe"

# API Endpoint
@app.post("/ingest")
async def ingest(event: Event):
    try:
        # Extract features
        features = extract_features(event)

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][prediction]

        # Decision
        decision = decide(prediction, probability, features[0])

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "decision": decision
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
