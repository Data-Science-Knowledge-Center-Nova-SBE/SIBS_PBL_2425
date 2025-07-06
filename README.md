# Fraud Detection Pipeline

A comprehensive, modular pipeline for fraud detection model training and inference. This pipeline is designed to be easy to run, maintain, and extend.

## 🚀 Features

- **Modular Architecture**: Clean separation between training and inference pipelines
- **Reusable Feature Engineering**: Centralized feature generation functions used by both pipelines
- **Multiple Aggregation Levels**: Daily and monthly prediction aggregation
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Model Versioning**: Automatic model saving with metadata and versioning
- **Flexible Data Input**: Support for Spark tables, CSV files, and pandas DataFrames
- **Easy Configuration**: Centralized configuration management

## 📁 Project Structure

```
fraud_pipeline/
├── fraud_pipeline/
│   ├── __init__.py              # Main module exports
│   ├── config.py                # Configuration settings
│   ├── utils.py                 # Utility functions
│   ├── feature_engineering.py   # Feature generation functions
│   ├── training_pipeline.py     # Training pipeline
│   └── inference_pipeline.py    # Inference pipeline
├── models/                      # Saved models directory
├── predictions/                 # Predictions output directory
├── logs/                        # Log files directory
├── artifacts/                   # Other artifacts (feature importance, etc.)
├── run_training.py              # Training script
├── run_inference.py             # Inference script
└── README.md                    # This file
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd fraud_pipeline
```

2. **Install dependencies**:
```bash
pip install pandas numpy scikit-learn pyspark joblib
```

3. **Set up Spark environment** (if not already configured):
```bash
# Install Java (required for Spark)
sudo apt-get install openjdk-8-jdk

# Set environment variables
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export SPARK_HOME=/path/to/spark
export PATH=$PATH:$SPARK_HOME/bin
```

## 📊 Usage

### Training Pipeline

The training pipeline handles data loading, feature engineering, model training, and evaluation.

#### Command Line Usage

```bash
# Basic training
python run_training.py

# With custom log level
python run_training.py --log-level DEBUG
```

#### Programmatic Usage

```python
from fraud_pipeline import FraudTrainingPipeline

# Initialize and run training pipeline
pipeline = FraudTrainingPipeline(log_level="INFO")
results = pipeline.run_full_pipeline()

print(f"Model saved to: {results['model_path']}")
print(f"Training samples: {results['training_samples']}")
print(f"Evaluation metrics: {results['evaluation_metrics']}")
```

### Inference Pipeline

The inference pipeline loads a trained model and makes predictions on new data.

#### Command Line Usage

```bash
# Basic inference (uses latest model and default data source)
python run_inference.py

# With specific data source
python run_inference.py --data-source "nova_sbe.new_data_table"

# With specific model
python run_inference.py --model-path "models/fraud_detection_model_20250101_120000.joblib"

# Don't save results to files
python run_inference.py --no-save
```

#### Programmatic Usage

```python
from fraud_pipeline import FraudInferencePipeline, run_inference_from_dataframe
import pandas as pd

# Method 1: Using pipeline class
pipeline = FraudInferencePipeline(model_path="path/to/model.joblib")
results = pipeline.run_inference_pipeline(data_source="table_name")

# Method 2: Using convenience function with DataFrame
df = pd.read_csv("new_data.csv")
results = run_inference_from_dataframe(df, save_results=True)

# Method 3: Using convenience function with table
from fraud_pipeline import run_inference_from_table
results = run_inference_from_table("nova_sbe.new_data", save_results=True)
```

## 🔧 Configuration

All configuration is centralized in `fraud_pipeline/config.py`:

### Key Configuration Sections

- **SPARK_CONFIG**: Spark session configuration
- **DATA_CONFIG**: Data source table names
- **MODEL_CONFIG**: Model hyperparameters and training settings
- **FEATURE_CONFIG**: Feature selection and target variable settings
- **PIPELINE_CONFIG**: Training/test date ranges and prediction horizons

### Example Configuration Update

```python
# In config.py
PIPELINE_CONFIG = {
    "training_start_date": "2023-06-01",
    "training_end_date": "2024-12-31",
    "test_start_date": "2025-01-01", 
    "test_end_date": "2025-04-30",
    "prediction_horizons": [1, 7, 30],
    "aggregation_levels": ["daily", "monthly"]
}
```

## 📈 Features Generated

The pipeline generates comprehensive features across multiple categories:

### Time Features
- Cyclical time features (sin/cos transformations)
- Weekend indicators
- Month, day, hour features

### Rolling Statistics (7 & 14-day windows)
- Rolling mean, max, min, sum, standard deviation
- Lag features (1, 7, 14 days)

### Merchant Features
- New merchant detection
- Merchant size classification
- MCC fraud risk ranking
- High-risk merchant identification

### Card Features
- Authorization response patterns
- Card cancellation tracking
- Remote transaction detection

### Daily Aggregations
- Transaction volumes and amounts
- Rejection rates and patterns
- Unique card/merchant counts

## 📊 Model Details

- **Algorithm**: Random Forest Regressor
- **Target Variable**: `fraud_amount_accepted` (corrected for labeling delay)
- **Feature Selection**: Based on Mutual Information and SHAP analysis
- **Hyperparameter Tuning**: Grid Search with Cross-Validation
- **Evaluation Metrics**: MAE, MAPE, Penalized MAE for multiple horizons

## 📁 Output Files

### Training Pipeline Outputs
- **Model File**: `models/fraud_detection_model_YYYYMMDD_HHMMSS.joblib`
- **Metadata**: `models/fraud_detection_model_YYYYMMDD_HHMMSS_metadata.json`
- **Feature Importance**: `artifacts/feature_importance_YYYYMMDD_HHMMSS.csv`
- **Logs**: `logs/fraud_pipeline_YYYYMMDD_HHMMSS.log`

### Inference Pipeline Outputs
- **Daily Predictions**: `predictions/fraud_amount_daily_predictions_YYYYMMDD_HHMMSS.csv`
- **Monthly Predictions**: `predictions/fraud_amount_monthly_predictions_YYYYMMDD_HHMMSS.csv`
- **Logs**: `logs/fraud_pipeline_YYYYMMDD_HHMMSS.log`

## 🔍 Example Workflows

### 1. Initial Model Training

```bash
# Train the initial model
python run_training.py --log-level INFO

# Check results
ls models/
ls artifacts/
```

### 2. Regular Model Retraining

```python
from fraud_pipeline import FraudTrainingPipeline

# Retrain with new data (update config.py first)
pipeline = FraudTrainingPipeline()
results = pipeline.run_full_pipeline()

print(f"New model performance: {results['evaluation_metrics']}")
```

### 3. Daily Prediction Generation

```bash
# Generate daily predictions
python run_inference.py --data-source "nova_sbe.latest_data"

# Check predictions
ls predictions/
```

### 4. Batch Processing with Custom Data

```python
import pandas as pd
from fraud_pipeline import run_inference_from_dataframe

# Load your data
df = pd.read_csv("custom_data.csv")

# Generate predictions
results = run_inference_from_dataframe(df, save_results=True)

# Access predictions
daily_preds = results['predictions']['daily']
monthly_preds = results['predictions']['monthly']
```

## 🚨 Troubleshooting

### Common Issues

1. **Spark Session Issues**:
   - Ensure Java is installed and JAVA_HOME is set
   - Check Spark configuration in config.py
   - Verify table access permissions

2. **Memory Issues**:
   - Adjust Spark memory settings in config.py
   - Reduce data size or increase cluster resources

3. **Missing Dependencies**:
   - Install required packages: `pip install -r requirements.txt`
   - Ensure findspark is available if using local Spark

4. **Model Not Found**:
   - Train a model first using the training pipeline
   - Check model path in models/ directory

### Logging

All operations are logged with timestamps and details. Check log files in the `logs/` directory for debugging information.

## 🔄 Extending the Pipeline

### Adding New Features

1. **Add feature functions** in `feature_engineering.py`
2. **Update configuration** in `config.py` to include new features
3. **Test with training pipeline** to ensure compatibility

### Adding New Models

1. **Modify training pipeline** to support new algorithms
2. **Update model configuration** in `config.py`
3. **Ensure inference pipeline** can load new model types

### Custom Aggregations

1. **Extend aggregation functions** in `utils.py`
2. **Update pipeline configuration** to include new levels
3. **Test with inference pipeline**

## 📞 Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Review configuration in `config.py`
3. Ensure all dependencies are installed
4. Check Spark cluster connectivity

## 🎯 Best Practices

1. **Regular Retraining**: Retrain models monthly or when performance degrades
2. **Monitor Predictions**: Track prediction accuracy and adjust thresholds
3. **Feature Monitoring**: Monitor feature distributions for data drift
4. **Backup Models**: Keep multiple model versions for rollback capability
5. **Log Analysis**: Regularly review logs for performance insights 