from fastapi.testclient import TestClient
from app import app
from unittest.mock import MagicMock
import app as app_module

# Mock the pipeline for tests
@app_module.load_pipeline = lambda: MagicMock(predict=lambda X: [1])

client = TestClient(app)

def test_ingest_approve():
    response = client.post("/ingest", json={"feature1": 5, "feature2": 3, "feature3": 2})
    assert response.status_code == 200
    assert response.json()["decision"] == "approve"

def test_ingest_reject():
    # Change mock to return 0
    app_module.load_pipeline = lambda: MagicMock(predict=lambda X: [0])
    response = client.post("/ingest", json={"feature1": 0, "feature2": 0, "feature3": 0})
    assert response.status_code == 200
    assert response.json()["decision"] == "reject"
