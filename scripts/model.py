import mlflow.pyfunc
import pandas as pd

def load_model_from_mlflow(run_id):
    model_uri = f"runs:/{run_id}/model"
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict(model, input_data):
    try:
        predictions = model.predict(pd.DataFrame(input_data, columns=["text"]))
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
