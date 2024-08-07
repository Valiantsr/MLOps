import mlflow.pyfunc
import pandas as pd

# Fungsi untuk memuat model dari MLflow
def load_model_from_mlflow(run_id):
    model_uri = f"runs:/{run_id}/model"
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Fungsi untuk melakukan prediksi
def predict(model, input_data):
    try:
        predictions = model.predict(pd.DataFrame(input_data, columns=["text"]))
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Ganti dengan run ID dari run yang telah Anda log
run_id = "d1cfd9dbdd0b4125a33a00b93704c014"

# Muat model
model = load_model_from_mlflow(run_id)

if model:
    # Contoh data input untuk prediksi
    input_data = ["I love this product!", "This is the worst experience I've ever had.", "It was okay, nothing special."]

    # Lakukan prediksi
    predictions = predict(model, input_data)

    if predictions is not None:
        for text, sentiment in zip(input_data, predictions):
            print(f"Text: {text} -> Sentiment: {sentiment}")
