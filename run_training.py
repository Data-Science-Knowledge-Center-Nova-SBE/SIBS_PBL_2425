#!/usr/bin/env python3
"""
Script to run the fraud detection training pipeline
"""

import sys
import argparse
from fraud_pipeline import FraudTrainingPipeline

def main():
    """Main function to run training pipeline"""
    parser = argparse.ArgumentParser(description="Run fraud detection training pipeline")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FRAUD DETECTION MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    try:
        # Initialize and run training pipeline
        pipeline = FraudTrainingPipeline(log_level=args.log_level)
        results = pipeline.run_full_pipeline()
        
        # Print results
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Status: {results['status']}")
        print(f"Model saved to: {results['model_path']}")
        print(f"Training samples: {results['training_samples']:,}")
        print(f"Test samples: {results['test_samples']:,}")
        print(f"Features used: {results['features_count']}")
        
        print("\nEvaluation Metrics:")
        for horizon, metrics in results['evaluation_metrics'].items():
            print(f"\n{horizon.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")
        
        print("\n" + "=" * 50)
        print("Training pipeline completed successfully!")
        print("You can now use the trained model for inference.")
        print("=" * 50)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Training pipeline failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 