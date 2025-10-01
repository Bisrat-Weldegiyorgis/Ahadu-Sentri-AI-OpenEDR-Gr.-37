from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Ahadu SentriAI - Threat Detection API",
              description="AI-powered security model for anomaly detection and response",
              version="1.0.0")

# Load the trained model and scaler
model = joblib.load("trained_model.pkl")
scaler = joblib.load('scaler.pkl')

app = FastAPI()




# Define the input data model with all features



@app.post("/ingest")
async def ingest(event: dict):
    features = extract_features(event)
    score = model.score(features)
    decision = decide(score, features)

# Convert the incoming data to a DataFrame


 # Preprocess the data


 # Make a prediction


  # Return the prediction and probability

def extract_features(event):
    # Example: extract numerical features from event dictionary
    return [event.get('feature1', 0), event.get('feature2', 0), event.get('feature3', 0)]

def decide(score, features):
    # Example: make a decision based on score and features
    if score > 0.8 and sum(features) > 10:
        return "approve"
    else:
        return "reject"

