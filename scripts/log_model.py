import mlflow
import mlflow.pyfunc
import os
import json
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging

# konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Gunakan variabel lingkungan untuk MLflow Tracking URI
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI","http://127.0.0.1:5000"))
logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

class CustomSentimentModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        logger.info("Loading model context")
        # Memuat semua artefak yang diperlukan
        self.config = json.load(open(context.artifacts["config"], encoding='utf-8'))
        self.special_tokens_map = json.load(open(context.artifacts["special_tokens_map"], encoding='utf-8'))
        self.tokenizer_config = json.load(open(context.artifacts["tokenizer_config"], encoding='utf-8'))
        with open(context.artifacts["vocab"], "r", encoding='utf-8') as f:
            self.vocab = f.read().splitlines()

        # Memuat tokenizer dan model dari file yang disimpan
        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["tokenizer_config"], local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(context.artifacts["model_file"], local_files_only=True)
        logger.info("Model context loaded succesfully")

    def predict(self, context, model_input):
        logger.info("Making predictions")
        inputs = self.tokenizer(model_input["text"].tolist(), return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        labels = ["negative", "neutral", "positive"]  # Sesuaikan dengan label Anda
        return [labels[pred] for pred in predictions]

def log_model():
    # Contoh input untuk model
    input_example = pd.DataFrame({"text": ["Aku suka produk ini", "Ini merupakan ayanan yang buruk"]})
    with mlflow.start_run() as run:
        logger.info(f"Started MLflow run with ID: {run.info.run_id}")

        # Log artifacts
        artifact_paths = {
            "config": "models/config.json",
            "model_file": "models/model.safetensors",
            "special_tokens_map": "models/special_tokens_map.json",
            "tokenizer_config": "models/tokenizer_config.json",
            "vocab": "models/vocab.txt"
        }

        for artifact_name, artifact_path in artifact_paths.items():
            if not os.path.exists(artifact_path):
                logger.error(f"Artifact file not found: {artifact_path}")
                raise FileNotFoundError(f"Artifact file not found: {artifact_path}")
            mlflow.log_artifact(artifact_path, artifact_path=artifact_name)
            logger.info(f"Logged artifact: {artifact_name}")

        # Log the custom model
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CustomSentimentModel(),
            artifacts={name: os.path.abspath(path) for name, path in artifact_paths.items()},
            input_example=input_example
        )
        logger.info("Custom model logged successfully")

    logger.info(f"Model logged in run: {run.info.run_id}")
    return run.info.run_id

# Log the custom model
# with mlflow.start_run() as run:
#     logger.info(f"Started MLflow run with ID: {run.info.run_id}")
#     # Log artifacts
#     mlflow.log_artifact("models/config.json", artifact_path="config")
#     mlflow.log_artifact("models/model.safetensors", artifact_path="model_file")
#     mlflow.log_artifact("models/special_tokens_map.json", artifact_path="special_tokens_map")
#     mlflow.log_artifact("models/tokenizer_config.json", artifact_path="tokenizer_config")
#     mlflow.log_artifact("models/vocab.txt", artifact_path="vocab")
    
#     # Log the custom model
#     mlflow.pyfunc.log_model(
#         artifact_path="model",
#         python_model=CustomSentimentModel(),
#         artifacts={
#             "config": os.path.abspath("models/config.json"),
#             "model_file": os.path.abspath("models/model.safetensors"),
#             "special_tokens_map": os.path.abspath("models/special_tokens_map.json"),
#             "tokenizer_config": os.path.abspath("models/tokenizer_config.json"),
#             "vocab": os.path.abspath("models/vocab.txt")
#         },
#         input_example=input_example
#     )

# print(f"Model logged in run: {run.info.run_id}")

if __name__ == "__main__":
    try:
        run_id = log_model()
        # Simpan run ID ke file
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        logger.info(f"Run ID saved to run_id.txt: {run_id}")
        print(f"MLFLOW_RUN_ID={run_id}")  # Untuk GitHub Actions
    except Exception as e:
        logger.error(f"Error during model logging: {e}")
        raise