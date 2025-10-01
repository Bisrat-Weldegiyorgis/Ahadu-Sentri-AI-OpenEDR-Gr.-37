import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_ingest_approve():
    # Example input that should trigger 'approve'
    test_event = {
        "feature1": 5,
        "feature2": 4,
        "feature3": 3
    }

    response = client.post("/ingest", json=test_event)
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert data["decision"] in ["approve", "reject"]  # depending on your pipeline logic

def test_ingest_reject():
    # Example input that should trigger 'reject'
    test_event = {
        "feature1": 0,
        "feature2": 0,
        "feature3": 0
    }

    response = client.post("/ingest", json=test_event)
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert data["decision"] in ["approve", "reject"]

def test_ingest_missing_features():
    # Event missing some features
    test_event = {}

    response = client.post("/ingest", json=test_event)
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
