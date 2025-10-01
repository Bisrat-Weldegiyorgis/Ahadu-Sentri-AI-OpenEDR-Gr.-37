from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Ahadu SentriAI" in response.json()["message"]

def test_predict_valid():
    payload = {
        "feature1": 0.5,
        "feature2": 1.2,
        "feature3": -0.3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)

def test_predict_invalid():
    # Missing feature3 should fail
    payload = {
        "feature1": 0.5,
        "feature2": 1.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
