# test_file.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import app as app_module  # Import your FastAPI app

# Mock the pipeline to avoid loading the real model
app_module.load_pipeline = lambda: MagicMock(predict=lambda X: [1])  # Always predicts "approve"

client = TestClient(app_module.app)

def test_ingest_approve():
    # Example input that should trigger 'approve'
    test_event = {
        "feature1": 5,
        "feature2": 4,
        "feature3": 3
    }

    response = client.post("/ingest", json=test_event)
    assert response.status_code == 200
    json_response = response.json()
    assert "decision" in json_response
    assert json_response["decision"] == "approve"

def test_ingest_reject():
    # Mock the predict to return 0 for rejection
    app_module.load_pipeline = lambda: MagicMock(predict=lambda X: [0])

    test_event = {
        "feature1": 0,
        "feature2": 0,
        "feature3": 0
    }

    response = client.post("/ingest", json=test_event)
    assert response.status_code == 200
    json_response = response.json()
    assert "decision" in json_response
    assert json_response["decision"] == "reject"

def test_ingest_missing_features():
    # Test with missing features, should still work due to default 0s in extract_features
    app_module.load_pipeline = lambda: MagicMock(predict=lambda X: [1])

    test_event = {}  # No features
    response = client.post("/ingest", json=test_event)
    assert response.status_code == 200
    json_response = response.json()
    assert "decision" in json_response
