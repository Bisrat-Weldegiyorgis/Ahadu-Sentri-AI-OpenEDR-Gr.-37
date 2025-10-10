import json
import time
import random
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

# Load trained model
# --------------------------
MODEL_PATH = Path("backend/random_forest_pipeline.pkl")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}. "
                            f"Make sure you exported it from Google Colab.")
    
rf_pipeline = joblib.load(MODEL_PATH)
print("✅ Random Forest pipeline loaded successfully and ready for prediction!")


# --------------------------
# Path to store detection results
# --------------------------
DATA_FILE = Path("data/detections.json")
DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
if not DATA_FILE.exists():
    with open(DATA_FILE, "w") as f:
        json.dump([], f)

# --------------------------
# Function to simulate live detections
# --------------------------
def generate_random_detection():
    # Example: 4 dummy feature values (adjust length to your model input)
    features = np.random.rand(1, rf_pipeline.named_steps['preprocessor'].transformers_[0][2].__len__() + 42)
    prediction = rf_pipeline.predict(features)[0]

    confidence = float(max(rf_pipeline.predict_proba(features)[0]))
    action = (
        "BLOCK" if confidence > 0.8 and prediction != "benign"
        else "QUARANTINE" if 0.5 < confidence <= 0.8
        else "ALLOW"
    )

    detection = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pattern_id": random.randint(1000, 9999),
        "prediction": prediction,
        "confidence": round(confidence, 3),
        "action": action
    }
    return detection

# --------------------------
# Append detection to history
# --------------------------
def save_detection(detection):
    with open(DATA_FILE, "r+") as f:
        data = json.load(f)
        data.append(detection)
        f.seek(0)
        json.dump(data, f, indent=2)

# --------------------------
# Main streamer loop
# --------------------------
def stream_detections(interval=3):
    print("🚀 Streaming detections... Press Ctrl+C to stop.")
    while True:
        detection = generate_random_detection()
        save_detection(detection)
        print(f"[+] New detection added: {detection}")
        time.sleep(interval)

if __name__ == "__main__":
    stream_detections()