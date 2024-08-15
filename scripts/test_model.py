import os
import mlflow
import pytest
import mlflow.pyfunc
import pandas as pd
from model import load_model_from_mlflow, predict
import logging
from mlflow.tracking import MlflowClient
import sys

# Run ID untuk testing
# test_run_id = "6e7c945be8c645278bc2dfda769b06ea"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mlflow_connection():
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()  # Mengganti list_experiments dengan search_experiments
        assert len(experiments) > 0, "No experiments found, check MLFLOW_TRACKING_URI"
        print(f"Successfully connected to MLflow with {len(experiments)} experiments found.")
    except Exception as e:
        assert False, f"Failed to connect to MLflow: {e}"
        print(f"Failed to connect to MLflow: {e}")

def load_model_from_mlflow(run_id=None):
    if not run_id:
        run_id = os.environ.get("MLFLOW_RUN_ID")
    
    if not run_id:
        raise ValueError("MLFLOW_RUN_ID not set in environment variables and not provided as argument")
    
    logger.info(f"Attempting to load model from run ID: {run_id}")
    
    try:
        model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        list_available_runs()
        raise

def list_available_runs():
    client = MlflowClient()
    logger.info("Available runs:")
    for run in client.search_runs(experiment_ids=['1']):  # Ganti '1' dengan ID eksperimen Anda jika berbeda
        logger.info(f"Run ID: {run.info.run_id}, Status: {run.info.status}")

def predict(model, input_data):
    logger.info("Making predictions")
    predictions = model.predict(pd.DataFrame({"text": input_data}))
    logger.info(f"Predictions: {predictions}")
    return predictions

def test_load_model_from_mlflow():
    try:
        model = load_model_from_mlflow()
        assert model is not None, "Model should be loaded successfully"
        logger.info("Model loaded successfully from MLflow.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        assert False, f"Error loading model: {e}"

def test_predict():
    try:
        model = load_model_from_mlflow()
        assert model is not None, "Model should be loaded successfully"
        logger.info("Model loaded successfully for prediction test.")

        input_data = [
            "I love this product!",
            "This is the worst experience I've ever had.",
            "It was okay, nothing special."
        ]

        predictions = predict(model, input_data)

        assert len(predictions) == len(input_data), f"Expected {len(input_data)} predictions, got {len(predictions)}"
        assert all(pred in ["positive", "negative", "neutral"] for pred in predictions), "Predictions should be 'positive', 'negative', or 'neutral'"
        
        logger.info(f"Predictions: {predictions}")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        assert False, f"Error during prediction: {e}"

if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    try:
        test_load_model_from_mlflow()
        test_predict()
        logger.info("All tests passed successfully")
    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)