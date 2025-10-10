from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("trained_model.pkl")
scaler = joblib.load("scaler (1).pkl")

# Define input schema
class InputData(BaseModel):
    age: int
    income: float
    gender: str

# Create FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "EDR system is live"}

@app.post("/predict")
def predict(data: InputData):
    # Convert input to array
    input_array = np.array([[data.feature1, data.feature2]])
    
    # Scale input
    scaled_input = scaler.transform(input_array)
    
    # Predict
    prediction = model.predict(scaled_input)
    
    return {"prediction": int(prediction[0])}
