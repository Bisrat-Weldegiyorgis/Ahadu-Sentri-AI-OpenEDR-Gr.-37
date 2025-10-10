from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, "trained_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

# Define the input data model with all features
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post("/predict/")
def predict(transaction: Transaction):
    try:
        # Convert the incoming data to a DataFrame
        data = pd.DataFrame([transaction.dict()])
        
        # Preprocess the data
        scaled_data = scaler.transform(data)
        
        # Make a prediction
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)
        
        # Return the prediction and probability
        return {"fraud_prediction": int(prediction[0]), "probability": float(prediction_proba[0][1])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
