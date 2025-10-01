from app import predict

def test_predict_output_type():
    result = predict([5.1, 3.5, 1.4, 0.2])
    assert isinstance(result, int) or isinstance(result, str)
