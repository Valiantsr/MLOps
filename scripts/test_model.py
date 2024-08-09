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
    # Muat model dari MLflow menggunakan run_id
    model = load_model_from_mlflow(test_run_id)
    assert model is not None, "Model should be loaded successfully"
    
    # Data input untuk prediksi
    input_data = ["I love this product!", "This is the worst experience I've ever had.", "It was okay, nothing special."]
    
    # Lakukan prediksi menggunakan model yang sebenarnya
    predictions = predict(model, input_data)
    
    # Ubah prediksi sesuai dengan format atau skala yang digunakan model Anda
    # Asumsikan model mengembalikan label sentimen seperti "positive", "negative", "neutral"
    
    assert predictions == ["positive", "negative", "neutral"], "Predictions should match expected results"
