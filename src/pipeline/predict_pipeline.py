import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    load_object,
    create_enhanced_features,
    identify_leakage_columns,
    load_config,
)


class PredictionPipeline:
    """
    Pipeline for generating predictions from trained models

    This pipeline loads trained models and preprocessors to make predictions:
    1. Churn probability
    2. Expected annual revenue
    3. Revenue at risk (churn prob × revenue)
    4. Customer segment assignment
    """

    def __init__(self, config_path=None):
        """
        Initialize prediction pipeline

        Args:
            config_path (str, optional): Path to configuration file
        """
        try:
            self.config_path = config_path
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if config_path:
                self.config = load_config(config_path)
            else:
                # Find latest config file
                config_dir = "config"
                if os.path.exists(config_dir):
                    config_files = [
                        f
                        for f in os.listdir(config_dir)
                        if f.startswith("config_") and f.endswith(".yaml")
                    ]
                    if config_files:
                        latest_config = max(config_files)
                        self.config_path = os.path.join(config_dir, latest_config)
                        self.config = load_config(self.config_path)
                    else:
                        raise FileNotFoundError(
                            "No configuration files found in config directory"
                        )
                else:
                    raise FileNotFoundError("Config directory not found")

            # Load model and preprocessor paths from config
            logging.info(f"Loading configuration from {self.config_path}")

            # Set model paths
            self.churn_model_path = self.config["model_trainer"]["churn_model_path"]
            self.revenue_model_path = self.config["model_trainer"]["revenue_model_path"]
            self.cluster_model_path = self.config["model_trainer"]["cluster_model_path"]

            # Set preprocessor paths
            self.preprocessor_path = self.config["data_transformation"][
                "preprocessor_obj_path"
            ]
            self.preprocessor_cluster_path = self.config["data_transformation"][
                "preprocessor_cluster_obj_path"
            ]
            self.preprocessor_reg_path = self.config["data_transformation"][
                "preprocessor_reg_obj_path"
            ]

            # Ensure all required files exist
            for path in [
                self.churn_model_path,
                self.revenue_model_path,
                self.cluster_model_path,
                self.preprocessor_path,
                self.preprocessor_cluster_path,
                self.preprocessor_reg_path,
            ]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Required file not found: {path}")

            # Load models and preprocessors
            self._load_components()

            logging.info("Prediction pipeline initialized successfully")

        except Exception as e:
            logging.error(f"Error initializing prediction pipeline: {e}")
            raise CustomException(e, sys)

    def _load_components(self):
        """Load all required models and preprocessors"""
        try:
            logging.info("Loading models and preprocessors")

            # Load models
            self.churn_model = load_object(self.churn_model_path)
            self.revenue_model = load_object(self.revenue_model_path)
            self.cluster_model = load_object(self.cluster_model_path)

            # Load preprocessors
            self.preprocessor = load_object(self.preprocessor_path)
            self.preprocessor_cluster = load_object(self.preprocessor_cluster_path)
            self.preprocessor_reg = load_object(self.preprocessor_reg_path)

            logging.info("Models and preprocessors loaded successfully")

        except Exception as e:
            logging.error(f"Error loading models and preprocessors: {e}")
            raise CustomException(e, sys)

    def predict(self, input_data):
        """
        Generate predictions using trained models

        Args:
            input_data (pd.DataFrame or str): Input data or path to CSV file

        Returns:
            pd.DataFrame: Predictions including churn probability, revenue,
                          revenue at risk, and cluster assignment
        """
        try:
            logging.info("Starting prediction process")

            # Handle input data (DataFrame or file path)
            if isinstance(input_data, str):
                # Load data from file
                logging.info(f"Loading input data from {input_data}")
                X = pd.read_csv(input_data)
            else:
                # Use provided DataFrame
                X = input_data.copy()

            logging.info(f"Input data shape: {X.shape}")

            # Apply feature engineering for churn prediction
            logging.info("Applying feature engineering")
            X_enhanced = create_enhanced_features(X)

            # Predict churn probability
            logging.info("Predicting churn probability")
            churn_proba = self.churn_model.predict_proba(X_enhanced)[:, 1]

            # Predict customer segment
            logging.info("Predicting customer segment")
            # Extract clustering features
            clustering_features = [
                "customer_age",
                "gender",
                "dependent_count",
                "months_on_book",
                "total_relationship_count",
                "months_inactive_12_mon",
                "contacts_count_12_mon",
                "credit_limit",
                "total_amt_chng_q4_q1",
                "total_ct_chng_q4_q1",
                "inactive_vs_tenure_ratio",
                "contacts_per_month",
                "product_penetration",
                "trans_count_change_ratio",
                "inactive_contacts",
                "tenure_product_ratio",
            ]

            # Make sure all required features are present
            missing_features = set(clustering_features) - set(X_enhanced.columns)
            if missing_features:
                raise ValueError(
                    f"Missing required features for clustering: {missing_features}"
                )

            X_cluster = X_enhanced[clustering_features].copy()
            X_cluster_processed = self.preprocessor_cluster.transform(X_cluster)
            cluster = self.cluster_model.predict(X_cluster_processed)

            # Prepare data for revenue prediction (remove leakage features)
            logging.info("Predicting annual revenue")
            leakage_columns = identify_leakage_columns([])
            X_revenue = X_enhanced.drop(
                columns=[col for col in leakage_columns if col in X_enhanced.columns]
            )
            revenue_pred = self.revenue_model.predict(X_revenue)

            # Calculate revenue at risk
            revenue_at_risk = churn_proba * revenue_pred

            # Create risk category
            risk_category = np.where(churn_proba > 0.5, "High Risk", "Low Risk")

            # Create value segments based on predicted revenue
            value_labels = [
                "Low Value",
                "Medium-Low Value",
                "Medium-High Value",
                "High Value",
            ]
            value_segment = pd.qcut(revenue_pred, 4, labels=value_labels)

            # Combine results
            predictions = pd.DataFrame(
                {
                    "churn_probability": churn_proba,
                    "cluster": cluster,
                    "estimated_annual_revenue": revenue_pred,
                    "revenue_at_risk": revenue_at_risk,
                    "risk_category": risk_category,
                    "value_segment": value_segment,
                }
            )

            # Add original input data
            result = pd.concat([X.reset_index(drop=True), predictions], axis=1)

            logging.info(f"Prediction completed. Output shape: {result.shape}")

            return result

        except Exception as e:
            logging.error(f"Error in prediction pipeline: {e}")
            raise CustomException(e, sys)

    def save_predictions(self, predictions, output_path=None):
        """
        Save predictions to file

        Args:
            predictions (pd.DataFrame): Prediction results
            output_path (str, optional): Path to save predictions

        Returns:
            str: Path where predictions were saved
        """
        try:
            if output_path is None:
                # Create default output path
                output_dir = os.path.join("artifacts", "predictions")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir, f"predictions_{self.timestamp}.csv"
                )

            # Save predictions
            predictions.to_csv(output_path, index=False)
            logging.info(f"Predictions saved to {output_path}")

            return output_path

        except Exception as e:
            logging.error(f"Error saving predictions: {e}")
            raise CustomException(e, sys)
