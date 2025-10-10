from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

sample_input = {
    "age": 28,
    "income": 45000.0,
    "gender": "female"
}

def test_predict_endpoint():
    response = client.post("/predict", json=sample_input)
    print(response.json())  # Optional: shows response for debugging
    assert response.status_code == 200
    assert "prediction" in response.json()
