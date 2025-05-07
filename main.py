import argparse
import os
import sys
from datetime import datetime

from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline
from src.exception import CustomException
from src.logger import logging


def main():
    """
    Main function to run the credit card customer analytics framework

    This script supports training and prediction modes through command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Credit Card Customer Analytics Framework"
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "predict"],
        help="Mode to run: train or predict",
    )

    # Optional arguments
    parser.add_argument("--data_path", type=str, help="Path to input data")
    parser.add_argument("--config_path", type=str, help="Path to configuration file")
    parser.add_argument(
        "--output_path", type=str, help="Path to save predictions (predict mode only)"
    )

    args = parser.parse_args()

    try:
        if args.mode == "train":
            logging.info("Running in training mode")

            # Initialize training pipeline
            pipeline = TrainingPipeline(config_path=args.config_path)

            # Run training
            results = pipeline.run(data_path=args.data_path)

            logging.info("Training completed successfully")
            print("\nTraining Summary:")
            print(
                f"Churn Model AUC: {results['churn_model']['metrics']['roc_auc']:.4f}"
            )
            print(
                f"Revenue Model RMSE: ${results['revenue_model']['metrics']['rmse']:.2f}"
            )
            print(
                f"Total Revenue at Risk: ${results['integrated_model']['metrics']['total_revenue_at_risk']:,.2f}"
            )
            print(
                f"Visualizations saved to: {results['churn_model']['visualizations'][0].rsplit('/', 1)[0]}"
            )

        elif args.mode == "predict":
            if not args.data_path:
                raise ValueError("--data_path is required for predict mode")

            logging.info("Running in prediction mode")

            # Initialize prediction pipeline
            pipeline = PredictionPipeline(config_path=args.config_path)

            # Generate predictions
            predictions = pipeline.predict(args.data_path)

            # Save predictions
            output_path = pipeline.save_predictions(predictions, args.output_path)

            logging.info("Prediction completed successfully")
            print("\nPrediction Summary:")
            print(f"Processed {len(predictions)} customer records")
            print(
                f"High Risk Customers: {(predictions['risk_category'] == 'High Risk').sum()} ({(predictions['risk_category'] == 'High Risk').mean() * 100:.2f}%)"
            )
            print(
                f"Total Revenue at Risk: ${predictions['revenue_at_risk'].sum():,.2f}"
            )
            print(f"Predictions saved to: {output_path}")

        else:
            raise ValueError(f"Invalid mode: {args.mode}")

    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
