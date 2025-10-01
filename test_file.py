import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import app as app_module

client = TestClient(app_module.app)

# Mock the pipeline for tests
app_module.pipeline = MagicMock()
app_module.pipeline.predict = MagicMock(side_effect=lambda X: [1])

def test_ingest_approve():
    test_event = {"feature1": 5, "feature2": 4, "feature3": 3}
    response = client.post("/ingest", json=test_event)
    assert response.status_code == 200
    assert response.json()["decision"] == "approve"

def test_ingest_reject():
    app_module.pipeline.predict = MagicMock(side_effect=lambda X: [0])
    test_event = {"feature1": 0, "feature2": 0, "feature3": 0}
    response = client.post("/ingest", json=test_event)
    assert response.status_code == 200
    assert response.json()["decision"] == "reject"

def test_ingest_missing_features():
    # Send empty dict; Pydantic defaults to 0
    test_event = {}
    response = client.post("/ingest", json=test_event)
    assert response.status_code == 200
    assert response.json()["decision"] == "approve"
