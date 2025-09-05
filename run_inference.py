#!/usr/bin/env python3
"""
Script to run the fraud detection inference pipeline
"""

import sys
import argparse
import pandas as pd
from fraud_pipeline import FraudInferencePipeline, run_inference_from_dataframe, run_inference_from_table

def main():
    """Main function to run inference pipeline"""
    parser = argparse.ArgumentParser(description="Run fraud detection inference pipeline")
    parser.add_argument("--data-source", type=str, 
                       help="Data source (table name or CSV file path)")
    parser.add_argument("--model-path", type=str, 
                       help="Path to trained model (if not specified, uses latest)")
    parser.add_argument("--no-save", action="store_true", 
                       help="Don't save results to files")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FRAUD DETECTION MODEL INFERENCE PIPELINE")
    print("=" * 60)
    
    try:
        # Initialize and run inference pipeline
        pipeline = FraudInferencePipeline(model_path=args.model_path, log_level=args.log_level)
        results = pipeline.run_inference_pipeline(
            data_source=args.data_source,
            save_results=not args.no_save
        )
        
        # Print results
        print("\n" + "=" * 50)
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Status: {results['status']}")
        print(f"Model used: {results['model_used']}")
        print(f"Predictions generated: {results['prediction_count']:,}")
        print(f"Aggregation levels: {results['aggregation_levels']}")
        
        if results['saved_files']:
            print("\nSaved files:")
            for level, path in results['saved_files'].items():
                print(f"  {level.capitalize()}: {path}")
        
        # Show sample predictions
        if 'daily' in results['predictions']:
            daily_preds = results['predictions']['daily']
            print(f"\nSample daily predictions:")
            print(daily_preds.head(10))
            
            # Show summary statistics
            print(f"\nPrediction Summary:")
            print(f"  Total predicted fraud amount: {daily_preds['prediction'].sum():,.2f}")
            print(f"  Average daily prediction: {daily_preds['prediction'].mean():,.2f}")
            print(f"  Max daily prediction: {daily_preds['prediction'].max():,.2f}")
            print(f"  Min daily prediction: {daily_preds['prediction'].min():,.2f}")
        
        if 'monthly' in results['predictions']:
            monthly_preds = results['predictions']['monthly']
            print(f"\nMonthly aggregated predictions:")
            print(monthly_preds)
        
        print("\n" + "=" * 50)
        print("Inference pipeline completed successfully!")
        print("=" * 50)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Inference pipeline failed: {str(e)}")
        return 1

def run_inference_from_csv(csv_file: str, model_path: str = None, save_results: bool = True):
    """
    Convenience function to run inference from a CSV file
    
    Args:
        csv_file: Path to CSV file
        model_path: Path to trained model
        save_results: Whether to save results
    """
    print(f"Loading data from CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from CSV")
    
    results = run_inference_from_dataframe(df, model_path, save_results)
    
    print("Inference completed!")
    return results

if __name__ == "__main__":
    sys.exit(main()) 