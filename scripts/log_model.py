import mlflow
import mlflow.pyfunc
import os
import json
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class CustomSentimentModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # Memuat semua artefak yang diperlukan
        self.config = json.load(open(context.artifacts["config"], encoding='utf-8'))
        self.special_tokens_map = json.load(open(context.artifacts["special_tokens_map"], encoding='utf-8'))
        self.tokenizer_config = json.load(open(context.artifacts["tokenizer_config"], encoding='utf-8'))
        with open(context.artifacts["vocab"], "r", encoding='utf-8') as f:
            self.vocab = f.read().splitlines()

        # Memuat tokenizer dan model dari file yang disimpan
        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["tokenizer_config"], local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(context.artifacts["model_file"], local_files_only=True)

    def predict(self, context, model_input):
        inputs = self.tokenizer(model_input["text"].tolist(), return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        labels = ["negative", "neutral", "positive"]  # Sesuaikan dengan label Anda
        return [labels[pred] for pred in predictions]

# Contoh input untuk model
input_example = pd.DataFrame({"text": ["I love this product!", "This is the worst experience I've ever had."]})

# Log the custom model
with mlflow.start_run() as run:
    # Log artifacts
    mlflow.log_artifact("models/config.json", artifact_path="config")
    mlflow.log_artifact("models/model.safetensors", artifact_path="model_file")
    mlflow.log_artifact("models/special_tokens_map.json", artifact_path="special_tokens_map")
    mlflow.log_artifact("models/tokenizer_config.json", artifact_path="tokenizer_config")
    mlflow.log_artifact("models/vocab.txt", artifact_path="vocab")
    
    # Log the custom model
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CustomSentimentModel(),
        artifacts={
            "config": os.path.abspath("models/config.json"),
            "model_file": os.path.abspath("models/model.safetensors"),
            "special_tokens_map": os.path.abspath("models/special_tokens_map.json"),
            "tokenizer_config": os.path.abspath("models/tokenizer_config.json"),
            "vocab": os.path.abspath("models/vocab.txt")
        },
        input_example=input_example
    )

print(f"Model logged in run: {run.info.run_id}")
