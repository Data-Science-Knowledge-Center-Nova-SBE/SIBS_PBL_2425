#!/usr/bin/env python3
"""
Example usage of the fraud detection pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fraud_pipeline import (
    FraudTrainingPipeline, 
    FraudInferencePipeline,
    run_inference_from_dataframe
)

def create_sample_data():
    """Create sample data for demonstration"""
    print("Creating sample data...")
    
    # Generate sample transaction data
    np.random.seed(42)
    n_rows = 1000
    
    # Create date range
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(30)]
    
    # Generate sample data
    data = []
    for date in dates:
        n_transactions = np.random.randint(20, 50)
        for _ in range(n_transactions):
            data.append({
                'TRANSACTION_DATETIME': date + timedelta(
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                ),
                'OPERATION_AMOUNT': np.random.lognormal(3, 1),
                'MERCHANT_NAME': np.random.choice(['Amazon', 'Walmart', 'Target', 'Costco', 'Best Buy']),
                'MCC': np.random.choice(['5411', '5999', '4829', '6051', '7995']),
                'PAN_ENCRYPTED': f'card_{np.random.randint(1000, 9999)}',
                'RESPONSE_STATUS_CDE': np.random.choice(['ACCP', 'RJCT'], p=[0.85, 0.15]),
                'AUTHORISATION_RESPONSE_CDE': np.random.choice([0, 14, 55, 61, 65, 82], p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.1]),
                'FRAUD_ACCEPTOR_CDE': np.random.choice(['REMOTE_ONLINE', 'POS_TERMINAL'], p=[0.6, 0.4]),
                'fraud_sp': np.random.choice([0, 1], p=[0.95, 0.05]),
                'fraud_classification_datetime': date + timedelta(days=np.random.randint(1, 30)),
                'id': f'tx_{len(data) + 1}'
            })
    
    df = pd.DataFrame(data)
    print(f"Created sample data with {len(df)} transactions")
    return df

def example_training_pipeline():
    """Example of running the training pipeline"""
    print("\n" + "="*50)
    print("EXAMPLE: TRAINING PIPELINE")
    print("="*50)
    
    try:
        # Note: This would normally use real Spark tables
        # For demonstration, we'll show the structure
        print("Training pipeline structure:")
        print("1. Load data from Spark tables")
        print("2. Engineer features")
        print("3. Split into train/test")
        print("4. Analyze feature importance")
        print("5. Train Random Forest model")
        print("6. Evaluate model performance")
        print("7. Save model and artifacts")
        
        # This is how you would actually run it:
        # pipeline = FraudTrainingPipeline(log_level="INFO")
        # results = pipeline.run_full_pipeline()
        # print(f"Model saved to: {results['model_path']}")
        
        print("\nTraining pipeline example completed!")
        
    except Exception as e:
        print(f"Training pipeline example failed: {e}")

def example_inference_pipeline():
    """Example of running the inference pipeline"""
    print("\n" + "="*50)
    print("EXAMPLE: INFERENCE PIPELINE")
    print("="*50)
    
    try:
        # Create sample data
        sample_df = create_sample_data()
        
        # Note: This would normally use a real trained model
        # For demonstration, we'll show the structure
        print("Inference pipeline structure:")
        print("1. Load trained model")
        print("2. Load new data")
        print("3. Engineer features")
        print("4. Make predictions")
        print("5. Aggregate predictions")
        print("6. Save results")
        
        # This is how you would actually run it:
        # results = run_inference_from_dataframe(sample_df, save_results=True)
        # print(f"Generated {results['prediction_count']} predictions")
        
        print("\nInference pipeline example completed!")
        
    except Exception as e:
        print(f"Inference pipeline example failed: {e}")

def example_configuration():
    """Example of configuration management"""
    print("\n" + "="*50)
    print("EXAMPLE: CONFIGURATION")
    print("="*50)
    
    print("Configuration is centralized in fraud_pipeline/config.py")
    print("\nKey configuration sections:")
    print("- SPARK_CONFIG: Spark session settings")
    print("- DATA_CONFIG: Data source table names")  
    print("- MODEL_CONFIG: Model hyperparameters")
    print("- FEATURE_CONFIG: Feature selection")
    print("- PIPELINE_CONFIG: Date ranges and horizons")
    
    print("\nExample configuration update:")
    print("""
# Update training date range
PIPELINE_CONFIG = {
    "training_start_date": "2023-06-01",
    "training_end_date": "2024-12-31",
    "test_start_date": "2025-01-01", 
    "test_end_date": "2025-04-30",
    "prediction_horizons": [1, 7, 30],
    "aggregation_levels": ["daily", "monthly"]
}
    """)

def example_file_structure():
    """Example of expected file structure"""
    print("\n" + "="*50)
    print("EXAMPLE: FILE STRUCTURE")
    print("="*50)
    
    print("After running the pipelines, you'll have:")
    print("""
fraud_pipeline/
├── models/
│   ├── fraud_detection_model_20250115_143022.joblib
│   └── fraud_detection_model_20250115_143022_metadata.json
├── predictions/
│   ├── fraud_amount_daily_predictions_20250115_143500.csv
│   └── fraud_amount_monthly_predictions_20250115_143500.csv
├── logs/
│   └── fraud_pipeline_20250115_143022.log
└── artifacts/
    └── feature_importance_20250115_143022.csv
    """)

def main():
    """Main function demonstrating the pipeline usage"""
    print("FRAUD DETECTION PIPELINE - USAGE EXAMPLES")
    print("="*60)
    
    # Show different examples
    example_training_pipeline()
    example_inference_pipeline()
    example_configuration()
    example_file_structure()
    
    print("\n" + "="*60)
    print("QUICK START COMMANDS")
    print("="*60)
    print("1. Train a model:")
    print("   python run_training.py")
    print("\n2. Make predictions:")
    print("   python run_inference.py")
    print("\n3. Use with custom data:")
    print("   python run_inference.py --data-source 'my_table'")
    print("\n4. Programmatic usage:")
    print("   from fraud_pipeline import FraudTrainingPipeline")
    print("   pipeline = FraudTrainingPipeline()")
    print("   results = pipeline.run_full_pipeline()")
    
    print("\n" + "="*60)
    print("For more details, see README.md")
    print("="*60)

if __name__ == "__main__":
    main() 