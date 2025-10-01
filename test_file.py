import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import app as app_module  # your FastAPI app module

client = TestClient(app_module.app)

@pytest.fixture(autouse=True)
def mock_pipeline():
    """
    Mock the model pipeline before each test.
    """
    # Ensure pipeline is mocked before any endpoint call
    app_module.pipeline = MagicMock()
    yield
    # Clean up after test
    app_module.pipeline = None


def test_ingest_approve():
    # Mock pipeline to return 1 → approve
    app_module.pipeline.predict = MagicMock(return_value=[1])

    test_event = {"feature1": 5, "feature2": 4, "feature3": 3}
    response = client.post("/ingest", json=test_event)
    
    assert response.status_code == 200
    assert response.json()["decision"] == "approve"
    assert response.json()["prediction"] == [1]


def test_ingest_reject():
    # Mock pipeline to return 0 → reject
    app_module.pipeline.predict = MagicMock(return_value=[0])

    test_event = {"feature1": 0, "feature2": 0, "feature3": 0}
    response = client.post("/ingest", json=test_event)
    
    assert response.status_code == 200
    assert response.json()["decision"] == "reject"
    assert response.json()["prediction"] == [0]


def test_ingest_missing_features():
    # Mock pipeline to return 1 → approve for empty input
    app_module.pipeline.predict = MagicMock(return_value=[1])

    test_event = {}  # No features provided
    response = client.post("/ingest", json=test_event)
    
    assert response.status_code == 200
    assert response.json()["decision"] == "approve"
    assert response.json()["prediction"] == [1]
