from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

class NetworkTraffic(BaseModel):
    duration: int
    protocol_type: str
    service: str
    flag: str
    src_bytes: int
    dst_bytes: int
    land: int = 0
    wrong_fragment: int = 0
    urgent: int = 0
    hot: int = 0
    num_failed_logins: int = 0
    logged_in: int = 0

app = FastAPI(title="Network Intrusion Detection API")

# Load model with better error handling
try:
    model = joblib.load('trained_model.pkl')
    preprocessor = joblib.load('fitted_preprocessor.pkl')
    print(" Model and preprocessor loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f" Error loading model: {e}")
    model = None
    preprocessor = None
    MODEL_LOADED = False

@app.get("/")
def read_root():
    return {"message": "Network Intrusion Detection API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": MODEL_LOADED}

@app.post("/predict/")
def predict(traffic: NetworkTraffic):
    if not MODEL_LOADED:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        input_data = pd.DataFrame([traffic.dict()])
        processed_data = preprocessor.transform(input_data)
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data).max()
        
        return {
            "prediction": prediction[0],
            "confidence": round(float(probability), 4),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
