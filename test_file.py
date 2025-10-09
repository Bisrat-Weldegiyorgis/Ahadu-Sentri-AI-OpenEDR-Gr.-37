import pytest
from fastapi.testclient import TestClient
from app import app  

client = TestClient(app)

def test_predict_approve():
    payload = {"feature1": 3.5, "feature2": 1.2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == "approve"

def test_predict_reject():
    payload = {"feature1": 0.0, "feature2": 0.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == "reject"

def test_missing_features():
    payload = {}  # Empty input
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity due to missing fields
