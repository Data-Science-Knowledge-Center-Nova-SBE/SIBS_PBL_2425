"""
Configuration file for fraud detection pipeline
"""
import os
from datetime import datetime, timedelta

# Spark Configuration
SPARK_CONFIG = {
    "python_packages_path": "/home/cdsw/ml_new.tar.gz#environment",
    "executor_memory": "10g",
    "driver_memory": "8g",
    "executor_cores": "3",
    "max_executors": 15,
    "memory_overhead": "2g",
    "n_partitions": 200,
    "driver_memory_overhead": 8000,
    "session_name": "fraud_pipeline"
}

# Data Configuration
DATA_CONFIG = {
    "training_table": "nova_sbe.raw_202306_202412",
    "card_status_table": "nova_sbe.card_status_202306_202412",
    "oos_table": "nova_sbe.sample_202412",
    "card_status_outsample_table": "nova_sbe.card_status_202501_202504"
}

# Model Configuration
MODEL_CONFIG = {
    "model_type": "RandomForestRegressor",
    "param_grid": {
        "n_estimators": [100, 300],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ['sqrt', 'log2']
    },
    "cv_folds": 3,
    "scoring": "neg_mean_absolute_error",
    "random_state": 42,
    "oversampling_threshold": 100000,
    "oversampling_factor": 5
}

# Feature Configuration
FEATURE_CONFIG = {
    "selected_features": [
        'unique_cards', 'total_rejected', 'total_accepted',
        'mcc_4829_count', 'total_operation_amount', 'total_tx', 
        'remote_tx', 'unique_merchant_card_pairs', 'new_merchant_tx', 
        'total_cancellations', 'month_sin', 'month_cos', 'is_weekend',
        'number_tx_exceeded', 'cash_withdrawal_exceeded', 'card_number_invalid',
        'invalid_pin', 'invalid_cvv', 'rejected_accepted_ratio', 'remote_ratio',
        'amount_transaction_ratio', 'avg_transaction_per_merchant',
        'rolling_mean_7', 'rolling_max_7', 'rolling_min_7', 'rolling_sum_7',
        'rolling_std_7', 'rolling_mean_14', 'rolling_max_14', 'rolling_min_14',
        'rolling_sum_14', 'rolling_std_14', 'lag_1', 'lag_7', 'lag_14',
        'day_of_month_cos', 'day_of_month_sin', 'mcc_6051_amount',
        'mcc_4829_amount', 'small_merchant_amount', 'big_merchant_amount',
        'new_merchant_amount', 'mcc_7995_amount', 'mcc_5999_amount',
        'mcc_4722_amount', 'penalized_total_operation_amount',
        'avg_pen_amount_per_card', 'avg_pen_amount_norm'
    ],
    "target_column": "fraud_amount_accepted",
    "corrected_target_column": "corrected_accepted_fraud_amount",
    "date_column": "TRANSACTION_DATE"
}

# Pipeline Configuration
PIPELINE_CONFIG = {
    "training_start_date": "2023-06-01",
    "training_end_date": "2024-12-31",
    "test_start_date": "2025-01-01",
    "test_end_date": "2025-04-30",
    "prediction_horizons": [1, 7, 30],
    "aggregation_levels": ["daily", "monthly"]
}

# File Paths
PATHS = {
    "models_dir": "models/",
    "predictions_dir": "predictions/",
    "features_dir": "features/",
    "logs_dir": "logs/",
    "artifacts_dir": "artifacts/",
    "plots_dir": "plots/"
}

# Ensure directories exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# High-risk MCCs and merchants (from EDA)
HIGH_RISK_CONFIG = {
    "high_risk_mccs": ["6051", "4829", "7995", "5999", "4722", "5944"],
    "high_risk_merchants": ["BETANO PT", "binance.com", "bifinity", "vinted"],
    "fraud_threshold_percentile": 0.95,
    "low_card_threshold": 40
} 
