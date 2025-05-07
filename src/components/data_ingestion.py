import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_config


@dataclass
class DataIngestionConfig:
    """Configuration class for data ingestion process"""

    raw_data_path: str = os.path.join("artifacts", "data", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "data", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data", "test.csv")
    config_path: str = os.path.join("config", "config.yaml")


class DataIngestion:
    """Class responsible for data loading, splitting, and saving"""

    def __init__(self, config_path=None):
        """
        Initialize DataIngestion with configuration

        Args:
            config_path (str, optional): Path to configuration file
        """
        self.ingestion_config = DataIngestionConfig()

        # Override with custom config if provided
        if config_path:
            self.ingestion_config.config_path = config_path
            custom_config = load_config(config_path)

            if "data_ingestion" in custom_config:
                data_config = custom_config["data_ingestion"]
                self.ingestion_config.raw_data_path = data_config.get(
                    "raw_data_path", self.ingestion_config.raw_data_path
                )
                self.ingestion_config.train_data_path = data_config.get(
                    "train_data_path", self.ingestion_config.train_data_path
                )
                self.ingestion_config.test_data_path = data_config.get(
                    "test_data_path", self.ingestion_config.test_data_path
                )

    def initiate_data_ingestion(self, data_path=None):
        """
        Load data from source, split into train/test, and save to disk

        Args:
            data_path (str, optional): Path to source data

        Returns:
            tuple: Paths to train and test datasets
        """
        try:
            # Use config path if explicit path not provided
            if data_path is None:
                # Use the fixed data path directly instead of reading from config
                data_path = "data/processed/clean_bank_data.csv"
                logging.info(f"Using fixed data path: {data_path}")

            # Debug information
            logging.info(f"Current working directory: {os.getcwd()}")
            logging.info(f"Reading data from {data_path}")
            logging.info(f"Absolute path: {os.path.abspath(data_path)}")
            logging.info(f"Does path exist? {os.path.exists(data_path)}")

            # Read data from file
            logging.info(f"Reading data from file: {data_path}")
            df = pd.read_csv(data_path)

            # Create directory for storing raw data
            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True
            )

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_path}")

            logging.info("Splitting data into train and test sets")

            # Separate features and target
            drop_columns = [
                "churn_flag",
                "est_annual_fee_revenue",
                "est_annual_interchange_revenue",
                "est_annual_revenue",
            ]

            # Add optional columns if they exist
            optional_columns = [
                "behavior_segment",
                "engagement_level",
                "utilization_level",
                "utilization_bin",
                "revolving_bin",
                "revenue_quintile",
                "tenure_bin",
            ]

            # Only drop columns that exist in the dataframe
            drop_columns.extend([col for col in optional_columns if col in df.columns])

            # Drop columns and get feature matrix
            X = df.drop(columns=drop_columns)
            y = df["churn_flag"]
            y_revenue = df["est_annual_revenue"]

            # Split with stratification to maintain churn distribution
            X_train, X_test, y_train, y_test, y_revenue_train, y_revenue_test = (
                train_test_split(
                    X, y, y_revenue, test_size=0.2, random_state=42, stratify=y
                )
            )

            # Recombine for saving
            train_data = pd.concat([X_train, y_train, y_revenue_train], axis=1)
            test_data = pd.concat([X_test, y_test, y_revenue_test], axis=1)

            # Create directories and save splits
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info(f"Train data saved to {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved to {self.ingestion_config.test_data_path}")

            # Log dataset statistics
            logging.info(f"Training set shape: {X_train.shape}")
            logging.info(f"Test set shape: {X_test.shape}")
            logging.info(f"Churn rate in training set: {y_train.mean():.4f}")
            logging.info(f"Churn rate in test set: {y_test.mean():.4f}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.error(f"Exception occurred during data ingestion: {e}")
            raise CustomException(e, sys)
