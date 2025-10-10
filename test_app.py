from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Normal input (should not trigger anomaly)
normal_input = {
    "feature1": 0.5,
    "feature2": 1.0,
    "feature3": 0.8,
    "age": 30,
    "income": 50000.0,
    "gender": "male"
}

# Anomalous input (high feature values and low income)
anomalous_input = {
    "feature1": 3.0,
    "feature2": 2.5,
    "feature3": 1.0,
    "age": 25,
    "income": 500.0,
    "gender": "female"
}

def test_normal_input():
    response = client.post("/predict", json=normal_input)
    assert response.status_code == 200
    assert response.json()["anomaly_detected"] is False

def test_anomalous_input():
    response = client.post("/predict", json=anomalous_input)
    assert response.status_code == 200
    assert response.json()["anomaly_detected"] is True
