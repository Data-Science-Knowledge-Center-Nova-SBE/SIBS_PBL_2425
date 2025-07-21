"""
Inference pipeline for fraud detection model
"""
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Union
import logging

from .utils import (
    setup_logging, create_spark_session, load_model, get_latest_model,
    save_predictions, aggregate_predictions
)
from .feature_engineering import generate_all_features
from .config import (
    DATA_CONFIG, FEATURE_CONFIG, PIPELINE_CONFIG, PATHS
)
from .plotting import plot_predictions

class FraudInferencePipeline:
    """
    Main inference pipeline for fraud detection
    """
    
    def __init__(self, model_path: Optional[str] = None, log_level: str = "INFO"):
        """
        Initialize the inference pipeline
        
        Args:
            model_path: Path to trained model (if None, uses latest model)
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)
        self.spark = create_spark_session("fraud_inference_pipeline")
        self.model = None
        self.model_path = model_path
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load the trained model"""
        if self.model_path is None:
            self.model_path = get_latest_model("fraud_detection_model")
            
        if self.model_path is None:
            raise ValueError("No trained model found. Please train a model first.")
            
        self.logger.info(f"Loading model from: {self.model_path}")
        self.model = load_model(self.model_path)
        self.logger.info("Model loaded successfully")
    
    def load_inference_data(self, data_source: Optional[str] = None) -> tuple:
        """
        Load data for inference
        
        Args:
            data_source: Optional data source (table name or file path)
            
        Returns:
            Tuple of (inference_df, training_df, card_status_df)
        """
        self.logger.info("Loading inference data...")
        
        if data_source:
            # Load from specified source
            if data_source.endswith('.csv'):
                inference_df = self.spark.read.csv(data_source, header=True, inferSchema=True)
            else:
                inference_df = self.spark.table(data_source)
        else:
            # Load default data sources
            inference_df = self.spark.table(DATA_CONFIG["oos_table"])
            
        # Load reference data for feature engineering
        training_df = self.spark.table(DATA_CONFIG["training_table"])
        card_status_df = self.spark.table(DATA_CONFIG["card_status_table"])
        card_status_outsample_df = self.spark.table(DATA_CONFIG["card_status_outsample_table"])
        
        # Combine card status data
        combined_card_status_df = card_status_df.union(card_status_outsample_df)
        
        self.logger.info(f"Loaded inference data with {inference_df.count()} rows")
        
        return inference_df, training_df, combined_card_status_df
    
    def load_dataframe(self, df: pd.DataFrame) -> tuple:
        """
        Load data from pandas DataFrame
        
        Args:
            df: Input pandas DataFrame
            
        Returns:
            Tuple of (inference_df, training_df, card_status_df)
        """
        self.logger.info("Loading data from provided DataFrame...")
        
        # Convert pandas DataFrame to Spark DataFrame
        inference_df = self.spark.createDataFrame(df)
        
        # Load reference data for feature engineering
        training_df = self.spark.table(DATA_CONFIG["training_table"])
        card_status_df = self.spark.table(DATA_CONFIG["card_status_table"])
        card_status_outsample_df = self.spark.table(DATA_CONFIG["card_status_outsample_table"])
        
        # Combine card status data
        combined_card_status_df = card_status_df.union(card_status_outsample_df)
        
        self.logger.info(f"Loaded DataFrame with {len(df)} rows")
        
        return inference_df, training_df, combined_card_status_df
    
    def engineer_features(self, inference_df, training_df, card_status_df) -> pd.DataFrame:
        """
        Generate features for inference data
        
        Args:
            inference_df: Inference Spark DataFrame
            training_df: Training Spark DataFrame
            card_status_df: Card status Spark DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Engineering features for inference...")
        
        # Generate all features (without correction for inference)
        features_df = generate_all_features(
            df=inference_df,
            training_df=training_df,
            card_status_df=card_status_df,
            apply_correction=False
        )
        
        self.logger.info(f"Generated {len(features_df.columns)} features")
        
        return features_df
    
    def make_predictions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the trained model
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            DataFrame with predictions
        """
        self.logger.info("Making predictions...")
        
        # Prepare features
        X = features_df[FEATURE_CONFIG["selected_features"]]
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Create predictions DataFrame
        predictions_df = features_df[[FEATURE_CONFIG["date_column"]]].copy()
        predictions_df['prediction'] = predictions
        predictions_df['prediction_date'] = datetime.now()
        predictions_df['model_path'] = self.model_path
        
        self.logger.info(f"Generated predictions for {len(predictions_df)} days")
        
        return predictions_df
        
       
    
    def aggregate_predictions(self, predictions_df: pd.DataFrame, 
                            aggregation_levels: list = None) -> Dict[str, pd.DataFrame]:
        """
        Aggregate predictions to different time levels
        
        Args:
            predictions_df: DataFrame with daily predictions
            aggregation_levels: List of aggregation levels (default: ['daily', 'monthly'])
            
        Returns:
            Dictionary with aggregated predictions
        """
        if aggregation_levels is None:
            aggregation_levels = PIPELINE_CONFIG["aggregation_levels"]
            
        self.logger.info(f"Aggregating predictions to: {aggregation_levels}")
        
        aggregated_results = {}
        
        for level in aggregation_levels:
            if level == "daily":
                # Daily is already the base level
                aggregated_results[level] = predictions_df.copy()
            else:
                # Aggregate to the specified level
                aggregated = aggregate_predictions(
                    predictions_df, 
                    date_column=FEATURE_CONFIG["date_column"],
                    value_column="prediction",
                    aggregation_level=level
                )
                aggregated_results[level] = aggregated
                
        return aggregated_results
    
    def save_predictions(self, predictions_dict: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Save predictions to files
        
        Args:
            predictions_dict: Dictionary with predictions at different aggregation levels
            
        Returns:
            Dictionary with saved file paths
        """
        self.logger.info("Saving predictions to files...")
        
        saved_paths = {}
        
        for level, predictions_df in predictions_dict.items():
            file_path = save_predictions(
                predictions_df, 
                prediction_type="fraud_amount",
                aggregation_level=level
            )
            saved_paths[level] = file_path
            self.logger.info(f"Saved {level} predictions to: {file_path}")
            
        return saved_paths
    
    def run_inference_pipeline(self, 
                             data_source: Optional[str] = None,
                             input_dataframe: Optional[pd.DataFrame] = None,
                             save_results: bool = True) -> Dict[str, Any]:
        """
        Run the complete inference pipeline
        
        Args:
            data_source: Data source (table name or file path)
            input_dataframe: Input pandas DataFrame (alternative to data_source)
            save_results: Whether to save results to files
            
        Returns:
            Dictionary with inference results
        """
        self.logger.info("Starting inference pipeline...")
        
        try:
            # Step 1: Load data
            if input_dataframe is not None:
                inference_df, training_df, card_status_df = self.load_dataframe(input_dataframe)
            else:
                inference_df, training_df, card_status_df = self.load_inference_data(data_source)
            
            # Step 2: Engineer features
            features_df = self.engineer_features(inference_df, training_df, card_status_df)
            
            # Step 3: Make predictions
            predictions_df = self.make_predictions(features_df)

            # Step 3b: Plot predictions
            plot_file = None
            try:
                plots_dir = PATHS.get("plots_dir", os.path.join(PATHS["artifacts_dir"], "plots"))
                dates = pd.to_datetime(predictions_df[FEATURE_CONFIG["date_column"]])
                preds = predictions_df["prediction"]
                actuals = features_df.get(FEATURE_CONFIG.get("corrected_target_column", ""), preds)
            
                plot_file = plot_predictions(
                    dates=dates,
                    actuals=actuals,
                    preds=preds,
                    output_dir=plots_dir,
                    title="Predicted Fraud Amount (Inference)"
                )
                self.logger.info(f"Inference prediction plot saved to: {plot_file}")
            except Exception as e:
                self.logger.warning(f"Prediction plot failed: {str(e)}")
            
            # Step 4: Aggregate predictions
            aggregated_predictions = self.aggregate_predictions(predictions_df)
            
            # Step 5: Save predictions (if requested)
            saved_paths = {}
            if save_results:
                saved_paths = self.save_predictions(aggregated_predictions)
            
            # Prepare results
            results = {
                "status": "success",
                "model_used": self.model_path,
                "prediction_count": len(predictions_df),
                "aggregation_levels": list(aggregated_predictions.keys()),
                "predictions": aggregated_predictions,
                "saved_files": saved_paths if save_results else {},
                "prediction_plot": plot_file
            }
            
            self.logger.info("Inference pipeline completed successfully!")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Inference pipeline failed: {str(e)}")
            raise
        
        finally:
            # Clean up Spark session
            if self.spark:
                self.spark.stop()

def run_inference_from_dataframe(df: pd.DataFrame, 
                                model_path: Optional[str] = None,
                                save_results: bool = True) -> Dict[str, Any]:
    """
    Convenience function to run inference from a pandas DataFrame
    
    Args:
        df: Input pandas DataFrame
        model_path: Path to trained model (if None, uses latest)
        save_results: Whether to save results to files
        
    Returns:
        Dictionary with inference results
    """
    pipeline = FraudInferencePipeline(model_path=model_path)
    return pipeline.run_inference_pipeline(input_dataframe=df, save_results=save_results)

def run_inference_from_table(table_name: str,
                           model_path: Optional[str] = None,
                           save_results: bool = True) -> Dict[str, Any]:
    """
    Convenience function to run inference from a Spark table
    
    Args:
        table_name: Name of the Spark table
        model_path: Path to trained model (if None, uses latest)
        save_results: Whether to save results to files
        
    Returns:
        Dictionary with inference results
    """
    pipeline = FraudInferencePipeline(model_path=model_path)
    return pipeline.run_inference_pipeline(data_source=table_name, save_results=save_results)

def main():
    """
    Main function to run the inference pipeline
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run fraud detection inference pipeline")
    parser.add_argument("--data-source", type=str, help="Data source (table name or file path)")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")
    
    args = parser.parse_args()
    
    pipeline = FraudInferencePipeline(model_path=args.model_path)
    results = pipeline.run_inference_pipeline(
        data_source=args.data_source,
        save_results=not args.no_save
    )
    
    print("=" * 50)
    print("INFERENCE PIPELINE RESULTS")
    print("=" * 50)
    print(f"Status: {results['status']}")
    print(f"Model used: {results['model_used']}")
    print(f"Predictions generated: {results['prediction_count']}")
    print(f"Aggregation levels: {results['aggregation_levels']}")
    
    if results['saved_files']:
        print("\nSaved files:")
        for level, path in results['saved_files'].items():
            print(f"  {level}: {path}")
    
    # Show sample predictions
    if 'daily' in results['predictions']:
        daily_preds = results['predictions']['daily']
        print(f"\nSample daily predictions:")
        print(daily_preds.head())
        
    if results.get("prediction_plot"):
        print(f"\nPrediction plot saved at: {results['prediction_plot']}")    


if __name__ == "__main__":
    main() 
