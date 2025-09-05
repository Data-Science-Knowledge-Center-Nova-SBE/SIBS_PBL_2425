"""
Fraud Detection Pipeline

A comprehensive pipeline for fraud detection model training and inference.
"""

from .training_pipeline import FraudTrainingPipeline
from .inference_pipeline import FraudInferencePipeline, run_inference_from_dataframe, run_inference_from_table
from .feature_engineering import generate_all_features
from .utils import setup_logging, create_spark_session, save_model, load_model

__version__ = "1.0.0"
__author__ = "Fraud Detection Team"

__all__ = [
    "FraudTrainingPipeline",
    "FraudInferencePipeline", 
    "run_inference_from_dataframe",
    "run_inference_from_table",
    "generate_all_features",
    "setup_logging",
    "create_spark_session",
    "save_model",
    "load_model"
] 
