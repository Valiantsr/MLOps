from setuptools import setup, find_packages

setup(
    name="MLOps",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "mlflow",
        "pandas",
        "transformers",
        "safetensors",
        "torch",
        "pytest"
    ],
)
