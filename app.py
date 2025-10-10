from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os

# =========================
# TRAINING SETUP
# =========================

# Define dummy training data (for demo/training)
X_train = np.array([
    [5, 4, 3],
    [0, 0, 0],
    [2, 1, 3],
    [8, 9, 7]
])
y_train = np.array([1, 0, 1, 1])  # 1 = approve, 0 = reject

# Define and train a simple model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])

model.fit(X_train, y_train)

# Save model
MODEL_PATH = "model_pipeline.pkl"
joblib.dump(model, MODEL_PATH)

# =========================
# FASTAPI APP
# =========================

app = FastAPI(title="Ahadu SentriAI - OpenEDR Model API")

class Event(BaseModel):
    feature1: float = 0.0
    feature2: float = 0.0
    feature3: float = 0.0


def load_pipeline():
    """Load or initialize the model pipeline."""
    global pipeline
    if not os.path.exists(MODEL_PATH):
        joblib.dump(model, MODEL_PATH)
    if "pipeline" not in globals() or pipeline is None:
        pipeline = joblib.load(MODEL_PATH)
    return pipeline


def extract_features(event: Event):
    """Extract input features from event."""
    return np.array([[event.feature1, event.feature2, event.feature3]])


def decide(prediction):
    """Convert numeric prediction to human-readable decision."""
    return "approve" if int(prediction[0]) == 1 else "reject"


@app.post("/ingest")
async def ingest(event: Event):
    """Handle incoming event and return prediction."""
    model = load_pipeline()
    features = extract_features(event)
    prediction = model.predict(features)
    decision = decide(prediction)
    return {"decision": decision, "prediction": list(prediction)}


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "Ahadu SentriAI API is running 🚀"}
