import pytest
import mlflow.pyfunc
import pandas as pd
from model import load_model_from_mlflow, predict

# Dummy run ID for testing
test_run_id = "d1cfd9dbdd0b4125a33a00b93704c014"

def test_load_model_from_mlflow():
    model = load_model_from_mlflow(test_run_id)
    assert model is not None, "Model should be loaded successfully"

def test_predict():
    # Mock model and prediction
    class MockModel:
        def predict(self, df):
            return ["positive" if "love" in text else "negative" for text in df["text"]]

    model = MockModel()
    input_data = ["I love this product!", "This is the worst experience I've ever had.", "It was okay, nothing special."]
    
    predictions = predict(model, input_data)
    
    assert predictions == ["positive", "negative", "negative"], "Predictions should match expected results"
