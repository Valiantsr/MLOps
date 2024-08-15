import pytest
import mlflow.pyfunc
import pandas as pd
from model import load_model_from_mlflow, predict

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Run ID untuk testing
test_run_id = "6e7c945be8c645278bc2dfda769b06ea"

def test_mlflow_connection():
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()  # Mengganti list_experiments dengan search_experiments
        assert len(experiments) > 0, "No experiments found, check MLFLOW_TRACKING_URI"
        print(f"Successfully connected to MLflow with {len(experiments)} experiments found.")
    except Exception as e:
        assert False, f"Failed to connect to MLflow: {e}"
        print(f"Failed to connect to MLflow: {e}")

def test_load_model_from_mlflow():
    try:
        model = load_model_from_mlflow(test_run_id)
        assert model is not None, "Model should be loaded successfully"
        print("Model loaded successfully from MLflow.")
    except Exception as e:
        assert False, f"Error loading model: {e}"
        print(f"Error loading model: {e}")

def test_predict():
    try:
        # Muat model dari MLflow menggunakan run_id
        model = load_model_from_mlflow(test_run_id)
        assert model is not None, "Model should be loaded successfully"
        print("Model loaded successfully for prediction test.")
        
        # Data input untuk prediksi
        input_data = ["I love this product!", "This is the worst experience I've ever had.", "It was okay, nothing special."]
        
        # Lakukan prediksi menggunakan model yang sebenarnya
        predictions = predict(model, input_data)
        
        # Ubah prediksi sesuai dengan format atau skala yang digunakan model Anda
        # Asumsikan model mengembalikan label sentimen seperti "positive", "negative", "neutral"
        assert predictions == ["positive", "negative", "neutral"], "Predictions should match expected results"
        print(f"Predictions: {predictions}")
    except Exception as e:
        assert False, f"Error during prediction: {e}"
        print(f"Error during prediction: {e}")