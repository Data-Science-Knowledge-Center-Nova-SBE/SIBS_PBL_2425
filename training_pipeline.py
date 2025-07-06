"""
Training pipeline for fraud detection model
"""
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_regression
import logging
from typing import Dict, Any, Optional, Tuple

from .utils import (
    setup_logging, create_spark_session, validate_dataframe, 
    save_model, calculate_metrics, penalized_mae_function
)
from .feature_engineering import generate_all_features
from .config import (
    DATA_CONFIG, MODEL_CONFIG, FEATURE_CONFIG, 
    PIPELINE_CONFIG, PATHS
)

class FraudTrainingPipeline:
    """
    Main training pipeline for fraud detection
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the training pipeline
        
        Args:
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)
        self.spark = create_spark_session("fraud_training_pipeline")
        self.model = None
        self.feature_importance = None
        self.training_metrics = {}
        
    def load_data(self) -> Tuple[Any, Any, Any]:
        """
        Load training data from Spark tables
        
        Returns:
            Tuple of (training_df, card_status_df, combined_card_status_df)
        """
        self.logger.info("Loading training data...")
        
        # Load main datasets
        training_df = self.spark.table(DATA_CONFIG["training_table"])
        card_status_df = self.spark.table(DATA_CONFIG["card_status_table"])
        oos_df = self.spark.table(DATA_CONFIG["oos_table"])
        card_status_outsample_df = self.spark.table(DATA_CONFIG["card_status_outsample_table"])
        
        # Ensure OOS has same columns as training
        oos_df = oos_df[training_df.columns]
        
        # Combine card status dataframes
        combined_card_status_df = card_status_df.union(card_status_outsample_df)
        
        # Combine training and OOS data
        full_df = training_df.union(oos_df)
        
        self.logger.info(f"Loaded training data with {training_df.count()} rows")
        self.logger.info(f"Loaded OOS data with {oos_df.count()} rows")
        self.logger.info(f"Combined dataset has {full_df.count()} rows")
        
        return full_df, training_df, combined_card_status_df
    
    def engineer_features(self, full_df: Any, training_df: Any, card_status_df: Any) -> pd.DataFrame:
        """
        Generate all features for the dataset
        
        Args:
            full_df: Full dataset (training + OOS)
            training_df: Training dataset only
            card_status_df: Card status dataset
            
        Returns:
            DataFrame with all engineered features
        """
        self.logger.info("Starting feature engineering...")
        
        # Generate all features
        features_df = generate_all_features(
            df=full_df,
            training_df=training_df,
            card_status_df=card_status_df,
            apply_correction=True
        )
        
        self.logger.info(f"Generated {len(features_df.columns)} features")
        self.logger.info(f"Feature engineering completed with {len(features_df)} rows")
        
        return features_df
    
    def split_data(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets
        
        Args:
            features_df: DataFrame with all features
            
        Returns:
            Tuple of (training_data, test_data)
        """
        self.logger.info("Splitting data into training and test sets...")
        
        # Convert date column to datetime
        features_df['TRANSACTION_DATE'] = pd.to_datetime(features_df['TRANSACTION_DATE'])
        
        # Split based on configured dates
        train_data = features_df[
            (features_df['TRANSACTION_DATE'] >= PIPELINE_CONFIG["training_start_date"]) & 
            (features_df['TRANSACTION_DATE'] <= PIPELINE_CONFIG["training_end_date"])
        ]
        
        test_data = features_df[
            (features_df['TRANSACTION_DATE'] >= PIPELINE_CONFIG["test_start_date"]) & 
            (features_df['TRANSACTION_DATE'] <= PIPELINE_CONFIG["test_end_date"])
        ]
        
        self.logger.info(f"Training set: {len(train_data)} rows")
        self.logger.info(f"Test set: {len(test_data)} rows")
        
        return train_data, test_data
    
    def analyze_feature_importance(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze feature importance using mutual information
        
        Args:
            train_data: Training dataset
            
        Returns:
            DataFrame with feature importance scores
        """
        self.logger.info("Analyzing feature importance with Mutual Information...")
        
        # Prepare features and target
        X = train_data.drop(columns=[
            'TRANSACTION_DATE', 'fraud_amount_accepted', 
            'days_since_transaction', 'correction_factor', 
            'corrected_accepted_fraud_amount'
        ], errors='ignore').fillna(0)
        
        y = train_data['fraud_amount_accepted']
        
        # Calculate mutual information scores
        mi_scores = mutual_info_regression(X, y)
        mi_df = pd.DataFrame({
            'Feature': X.columns, 
            'MI_Score': mi_scores
        }).sort_values(by='MI_Score', ascending=False)
        
        self.feature_importance = mi_df
        
        # Save feature importance
        importance_path = f"{PATHS['artifacts_dir']}feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        mi_df.to_csv(importance_path, index=False)
        
        self.logger.info(f"Feature importance analysis completed. Saved to {importance_path}")
        self.logger.info(f"Top 10 features:\n{mi_df.head(10)}")
        
        return mi_df
    
    def prepare_training_data(self, train_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with oversampling
        
        Args:
            train_data: Training dataset
            
        Returns:
            Tuple of (X_train, y_train)
        """
        self.logger.info("Preparing training data with oversampling...")
        
        # Manual oversampling: duplicate high-fraud days
        threshold = MODEL_CONFIG["oversampling_threshold"]
        factor = MODEL_CONFIG["oversampling_factor"]
        
        peak_days = train_data[train_data["corrected_accepted_fraud_amount"] > threshold]
        normal_days = train_data[train_data["corrected_accepted_fraud_amount"] <= threshold]
        
        # Create balanced dataset
        balanced_data = pd.concat([normal_days] + [peak_days] * factor, ignore_index=True)
        
        # Prepare features and target
        X_train = balanced_data[FEATURE_CONFIG["selected_features"]]
        y_train = balanced_data[FEATURE_CONFIG["corrected_target_column"]]
        
        self.logger.info(f"Original training size: {len(train_data)}")
        self.logger.info(f"Balanced training size: {len(balanced_data)}")
        self.logger.info(f"Peak days duplicated: {len(peak_days)} x {factor}")
        
        return X_train, y_train
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
        """
        Train the fraud detection model
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        self.logger.info("Training Random Forest model with hyperparameter tuning...")
        
        # Initialize model
        rf = RandomForestRegressor(
            random_state=MODEL_CONFIG["random_state"],
            n_jobs=-1
        )
        
        # Grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            rf, 
            MODEL_CONFIG["param_grid"], 
            cv=MODEL_CONFIG["cv_folds"], 
            scoring=MODEL_CONFIG["scoring"],
            verbose=1,
            n_jobs=-1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the trained model
        
        Args:
            test_data: Test dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info("Evaluating model performance...")
        
        # Prepare test data
        X_test = test_data[FEATURE_CONFIG["selected_features"]]
        y_test = test_data[FEATURE_CONFIG["corrected_target_column"]]
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics for different horizons
        evaluation_results = {}
        
        for horizon in PIPELINE_CONFIG["prediction_horizons"]:
            if len(y_test) >= horizon:
                y_h = y_test.iloc[:horizon]
                p_h = pd.Series(y_pred[:horizon], index=y_h.index)
                
                metrics = calculate_metrics(y_h, p_h)
                metrics['penalized_mae'] = penalized_mae_function(y_h.values, p_h.values)
                
                evaluation_results[f"{horizon}_day"] = metrics
                
                self.logger.info(f"{horizon}-Day Metrics:")
                for metric, value in metrics.items():
                    self.logger.info(f"  {metric.upper()}: {value:.4f}")
        
        # Store evaluation results
        self.training_metrics = evaluation_results
        
        return evaluation_results
    
    def save_model_artifacts(self) -> str:
        """
        Save the trained model and related artifacts
        
        Returns:
            Path to saved model
        """
        self.logger.info("Saving model artifacts...")
        
        # Prepare metadata
        metadata = {
            "model_type": MODEL_CONFIG["model_type"],
            "training_date": datetime.now().isoformat(),
            "training_period": f"{PIPELINE_CONFIG['training_start_date']} to {PIPELINE_CONFIG['training_end_date']}",
            "test_period": f"{PIPELINE_CONFIG['test_start_date']} to {PIPELINE_CONFIG['test_end_date']}",
            "features_used": FEATURE_CONFIG["selected_features"],
            "model_parameters": self.model.get_params() if self.model else None,
            "evaluation_metrics": self.training_metrics,
            "feature_importance_top_10": self.feature_importance.head(10).to_dict() if self.feature_importance is not None else None
        }
        
        # Save model
        model_path = save_model(self.model, "fraud_detection_model", metadata)
        
        self.logger.info(f"Model saved to: {model_path}")
        
        return model_path
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("Starting full training pipeline...")
        
        try:
            # Step 1: Load data
            full_df, training_df, card_status_df = self.load_data()
            
            # Step 2: Engineer features
            features_df = self.engineer_features(full_df, training_df, card_status_df)
            
            # Step 3: Split data
            train_data, test_data = self.split_data(features_df)
            
            # Step 4: Analyze feature importance
            self.analyze_feature_importance(train_data)
            
            # Step 5: Prepare training data
            X_train, y_train = self.prepare_training_data(train_data)
            
            # Step 6: Train model
            self.train_model(X_train, y_train)
            
            # Step 7: Evaluate model
            evaluation_results = self.evaluate_model(test_data)
            
            # Step 8: Save model artifacts
            model_path = self.save_model_artifacts()
            
            # Prepare results
            results = {
                "status": "success",
                "model_path": model_path,
                "evaluation_metrics": evaluation_results,
                "training_samples": len(X_train),
                "test_samples": len(test_data),
                "features_count": len(FEATURE_CONFIG["selected_features"])
            }
            
            self.logger.info("Training pipeline completed successfully!")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            raise
        
        finally:
            # Clean up Spark session
            if self.spark:
                self.spark.stop()

def main():
    """
    Main function to run the training pipeline
    """
    pipeline = FraudTrainingPipeline()
    results = pipeline.run_full_pipeline()
    
    print("=" * 50)
    print("TRAINING PIPELINE RESULTS")
    print("=" * 50)
    print(f"Status: {results['status']}")
    print(f"Model saved to: {results['model_path']}")
    print(f"Training samples: {results['training_samples']}")
    print(f"Test samples: {results['test_samples']}")
    print(f"Features used: {results['features_count']}")
    print("\nEvaluation Metrics:")
    for horizon, metrics in results['evaluation_metrics'].items():
        print(f"\n{horizon.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main() 