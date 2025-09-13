from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Ahadu SentriAI - Threat Detection API",
              description="AI-powered security model for anomaly detection",
              version="1.0.0")

# Load the trained model and scaler
model = joblib.load("trained_model.ipynb")

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


from utils import decide, extract_features
import actions

def handle_event(event):
    features = extract_features(event)
    score = compute_score(features)  # assume this is defined
    decision = decide(score, features)

    if decision == "isolate":
        actions.isolate_endpoint(event)
    elif decision == "notify":
        actions.notify_admin(event, score)
    else:
        actions.allow(event)

    return {"score": score, "decision": decision}
