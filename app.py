from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np


# Define your input data model

class Event(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add more features as needed


# Initialize FastAPI app

app = FastAPI(
    title="Ahadu SentriAI - Threat Detection API",
    description="AI-powered security model for anomaly detection and response",
    version="1.0.0"
)


# Load trained pipeline

try:
    pipeline = joblib.load("trained_model.pkl")  # Make sure this file exists
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


# Prediction endpoint

@app.post("/predict")
async def predict(event: Event):
    try:
        # Extract features from request
        features = np.array([[
            event.feature1,
            event.feature2,
            event.feature3
            # add more features here in same order as training
        ]])
        
        # Predict using the pipeline (scaler + model)
        prediction = pipeline.predict(features)[0]
        probabilities = pipeline.predict_proba(features)[0].tolist()
        
        return {
            "prediction": int(prediction),
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")


# Health check endpoint

@app.get("/")
async def root():
    return {"message": "Ahadu SentriAI API is running!"}
