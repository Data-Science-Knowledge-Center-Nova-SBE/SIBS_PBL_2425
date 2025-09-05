"""
Utility functions for fraud detection pipeline
"""
import logging
import os
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import joblib
import json
from typing import Dict, Any, Optional, Union
from .config import SPARK_CONFIG, PATHS

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Logger instance
    """
    log_filename = os.path.join(PATHS["logs_dir"], f"fraud_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger

def create_spark_session(app_name: str = None) -> SparkSession:
    """
    Create and configure Spark session
    
    Args:
        app_name: Application name for Spark session
    
    Returns:
        SparkSession instance
    """
    if app_name is None:
        app_name = SPARK_CONFIG["session_name"]
    
    try:
        import findspark
        findspark.init()
    except ImportError:
        pass
    
    spark = SparkSession.builder \
        .config("spark.yarn.dist.archives", SPARK_CONFIG["python_packages_path"]) \
        .config("spark.executor.memory", SPARK_CONFIG["executor_memory"]) \
        .config("spark.driver.memory", SPARK_CONFIG["driver_memory"]) \
        .config("spark.executor.cores", SPARK_CONFIG["executor_cores"]) \
        .config("spark.yarn.executor.memoryOverhead", SPARK_CONFIG["memory_overhead"]) \
        .config("spark.dynamicAllocation.enabled", True) \
        .config("spark.dynamicAllocation.maxExecutors", SPARK_CONFIG["max_executors"]) \
        .config("spark.sql.shuffle.partitions", SPARK_CONFIG["n_partitions"]) \
        .config("spark.sql.adaptive.enabled", True) \
        .config("spark.yarn.driver.memoryOverhead", SPARK_CONFIG["driver_memory_overhead"]) \
        .config("spark.yarn.queue", "root.nova_sbe") \
        .appName(app_name) \
        .getOrCreate()
    
    return spark

def validate_dataframe(df: Union[pd.DataFrame, 'pyspark.sql.DataFrame'], 
                      required_columns: list, 
                      df_name: str = "DataFrame") -> bool:
    """
    Validate that a dataframe has required columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        df_name: Name of the dataframe for error messages
    
    Returns:
        True if valid, raises ValueError if not
    """
    if hasattr(df, 'columns'):  # pandas DataFrame
        df_columns = df.columns.tolist()
    else:  # PySpark DataFrame
        df_columns = df.columns
    
    missing_columns = set(required_columns) - set(df_columns)
    
    if missing_columns:
        raise ValueError(f"{df_name} is missing required columns: {missing_columns}")
    
    return True

def save_model(model: Any, model_name: str, metadata: Dict[str, Any] = None) -> str:
    """
    Save model and metadata
    
    Args:
        model: Trained model object
        model_name: Name for the model file
        metadata: Additional metadata to save
    
    Returns:
        Path to saved model
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"{model_name}_{timestamp}.joblib"
    model_path = os.path.join(PATHS["models_dir"], model_filename)
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save metadata
    if metadata:
        metadata_path = os.path.join(PATHS["models_dir"], f"{model_name}_{timestamp}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    return model_path

def load_model(model_path: str) -> Any:
    """
    Load saved model
    
    Args:
        model_path: Path to model file
    
    Returns:
        Loaded model object
    """
    return joblib.load(model_path)

def get_latest_model(model_name: str) -> Optional[str]:
    """
    Get path to the latest model file
    
    Args:
        model_name: Base name of the model
    
    Returns:
        Path to latest model file or None if not found
    """
    model_files = [f for f in os.listdir(PATHS["models_dir"]) 
                   if f.startswith(model_name) and f.endswith('.joblib')]
    
    if not model_files:
        return None
    
    # Sort by timestamp in filename
    model_files.sort(reverse=True)
    return os.path.join(PATHS["models_dir"], model_files[0])

def save_predictions(predictions: pd.DataFrame, 
                    prediction_type: str, 
                    aggregation_level: str = "daily") -> str:
    """
    Save predictions to file
    
    Args:
        predictions: DataFrame with predictions
        prediction_type: Type of prediction (e.g., 'fraud_amount')
        aggregation_level: Aggregation level (daily, monthly)
    
    Returns:
        Path to saved predictions file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{prediction_type}_{aggregation_level}_predictions_{timestamp}.csv"
    filepath = os.path.join(PATHS["predictions_dir"], filename)
    
    predictions.to_csv(filepath, index=False)
    return filepath

def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Calculate evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    
    return metrics

def penalized_mae_function(y_true: np.ndarray, y_pred: np.ndarray, 
                          penalty_factor: float = 2.0) -> float:
    """
    Calculate penalized MAE (higher penalty for underestimating fraud)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        penalty_factor: Penalty multiplier for underestimation
    
    Returns:
        Penalized MAE value
    """
    import numpy as np
    
    errors = y_true - y_pred
    penalties = np.where(errors > 0, penalty_factor, 1.0)  # Penalize underestimation
    penalized_errors = np.abs(errors) * penalties
    
    return np.mean(penalized_errors)

def aggregate_predictions(predictions: pd.DataFrame, 
                         date_column: str = "TRANSACTION_DATE",
                         value_column: str = "prediction",
                         aggregation_level: str = "monthly") -> pd.DataFrame:
    """
    Aggregate predictions to monthly level
    
    Args:
        predictions: DataFrame with daily predictions
        date_column: Name of date column
        value_column: Name of value column to aggregate
        aggregation_level: Level of aggregation (monthly, weekly)
    
    Returns:
        Aggregated DataFrame
    """
    predictions[date_column] = pd.to_datetime(predictions[date_column])
    
    if aggregation_level == "monthly":
        predictions['period'] = predictions[date_column].dt.to_period('M')
        aggregated = predictions.groupby('period').agg({
            value_column: ['sum', 'mean', 'std', 'min', 'max']
        }).reset_index()
        
        # Flatten column names
        aggregated.columns = ['period', 'total', 'mean', 'std', 'min', 'max']
        aggregated['period'] = aggregated['period'].astype(str)
        
    elif aggregation_level == "weekly":
        predictions['period'] = predictions[date_column].dt.to_period('W')
        aggregated = predictions.groupby('period').agg({
            value_column: ['sum', 'mean', 'std', 'min', 'max']
        }).reset_index()
        
        # Flatten column names
        aggregated.columns = ['period', 'total', 'mean', 'std', 'min', 'max']
        aggregated['period'] = aggregated['period'].astype(str)
    
    else:
        raise ValueError(f"Unsupported aggregation level: {aggregation_level}")
    
    return aggregated 