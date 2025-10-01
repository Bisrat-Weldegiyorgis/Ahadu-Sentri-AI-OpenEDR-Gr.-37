from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np


# Load trained pipeline

try:
    pipeline = joblib.load("trained_model.pkl")  # includes scaler + model
except Exception as e:
    raise RuntimeError(f" Failed to load model: {e}")


# Initialize FastAPI app

app = FastAPI(
    title="Ahadu SentriAI - Threat Detection API",
    description="AI-powered anomaly detection using ML pipeline (scaler + model).",
    version="1.0.0"
)


# Define Input Schema

class EventData(BaseModel):
    feature1: float
    feature2: float
    feature3: float


# Predict Endpoint

@app.post("/predict")
async def predict(data: EventData):
    try:
        # Convert input to array
        features = np.array([[data.feature1, data.feature2, data.feature3]])

        # Run through pipeline (scaler + model)
        prediction = pipeline.predict(features)[0]
        probability = pipeline.predict_proba(features).max()

        return {
            "prediction": int(prediction),
            "probability": float(probability)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": " Ahadu SentriAI API is running!"}
