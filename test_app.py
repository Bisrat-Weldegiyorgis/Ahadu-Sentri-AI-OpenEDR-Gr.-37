def test_prediction_endpoint():
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
