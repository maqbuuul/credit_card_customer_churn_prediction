import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    save_object,
    load_object,
    load_config,
    create_enhanced_features,
    identify_leakage_columns,
)


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation process"""

    preprocessor_obj_path: str = os.path.join(
        "artifacts", "preprocessor", "preprocessor.pkl"
    )
    preprocessor_cluster_obj_path: str = os.path.join(
        "artifacts", "preprocessor", "cluster_preprocessor.pkl"
    )
    preprocessor_reg_obj_path: str = os.path.join(
        "artifacts", "preprocessor", "reg_preprocessor.pkl"
    )
    pca_obj_path: str = os.path.join("artifacts", "preprocessor", "pca.pkl")
    config_path: str = os.path.join("config", "config.yaml")
    train_enhanced_data_path: str = os.path.join(
        "artifacts", "data", "train_enhanced.csv"
    )
    test_enhanced_data_path: str = os.path.join(
        "artifacts", "data", "test_enhanced.csv"
    )
    train_enhanced_cluster_data_path: str = os.path.join(
        "artifacts", "data", "train_enhanced_cluster.csv"
    )
    train_enhanced_reg_data_path: str = os.path.join(
        "artifacts", "data", "train_enhanced_reg.csv"
    )
    test_enhanced_reg_data_path: str = os.path.join(
        "artifacts", "data", "test_enhanced_reg.csv"
    )


class DataTransformation:
    """Class responsible for data preprocessing and transformation"""

    def __init__(self, config_path=None):
        """
        Initialize DataTransformation with configuration

        Args:
            config_path (str, optional): Path to configuration file
        """
        self.transformation_config = DataTransformationConfig()

        # Override with custom config if provided
        if config_path:
            self.transformation_config.config_path = config_path
            custom_config = load_config(config_path)

            if "data_transformation" in custom_config:
                transform_config = custom_config["data_transformation"]
                for key, value in transform_config.items():
                    if hasattr(self.transformation_config, key):
                        setattr(self.transformation_config, key, value)

    def get_data_transformer(self, num_features, cat_features):
        """
        Create a preprocessing pipeline for classification

        Args:
            num_features (list): Numerical feature column names
            cat_features (list): Categorical feature column names

        Returns:
            ColumnTransformer: Preprocessing pipeline for features
        """
        try:
            logging.info("Creating preprocessing pipeline for classification")

            # For numeric features: impute missing values and scale
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                ]
            )

            # For categorical features: impute missing values and one-hot encode
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )

            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, num_features),
                    ("cat", categorical_transformer, cat_features),
                ]
            )

            logging.info(
                f"Classification preprocessor created with {len(num_features)} numerical and {len(cat_features)} categorical features"
            )
            return preprocessor

        except Exception as e:
            logging.error(f"Error in get_data_transformer: {e}")
            raise CustomException(e, sys)

    def get_regression_transformer(self, num_features, cat_features):
        """
        Create a preprocessing pipeline for regression

        Args:
            num_features (list): Numerical feature column names
            cat_features (list): Categorical feature column names

        Returns:
            ColumnTransformer: Preprocessing pipeline for regression features
        """
        try:
            logging.info("Creating preprocessing pipeline for regression")

            # For numeric features: impute missing values and scale
            numeric_transformer_reg = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # For categorical features: impute missing values and one-hot encode
            categorical_transformer_reg = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )

            # Combine preprocessing steps
            preprocessor_reg = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer_reg, num_features),
                    ("cat", categorical_transformer_reg, cat_features),
                ]
            )

            logging.info(
                f"Regression preprocessor created with {len(num_features)} numerical and {len(cat_features)} categorical features"
            )
            return preprocessor_reg

        except Exception as e:
            logging.error(f"Error in get_regression_transformer: {e}")
            raise CustomException(e, sys)

    def get_cluster_transformer(self, num_features):
        """
        Create a preprocessing pipeline for clustering

        Args:
            num_features (list): Feature column names for clustering

        Returns:
            Pipeline: Preprocessing pipeline for clustering features
        """
        try:
            logging.info("Creating preprocessing pipeline for clustering")

            # For clustering, we handle numeric features with imputation and scaling
            preprocessor_cluster = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                ]
            )

            logging.info(
                f"Clustering preprocessor created with {len(num_features)} features"
            )
            return preprocessor_cluster

        except Exception as e:
            logging.error(f"Error in get_cluster_transformer: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Transform data for modeling by applying feature engineering and preprocessing

        Args:
            train_path (str): Path to training data
            test_path (str): Path to test data

        Returns:
            tuple: Paths to transformed data and preprocessor objects
        """
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train DataFrame Head: \n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head: \n{test_df.head().to_string()}")

            # Separate features and targets
            X_train = train_df.drop(columns=["churn_flag", "est_annual_revenue"])
            y_train = train_df["churn_flag"]
            y_revenue_train = train_df["est_annual_revenue"]

            X_test = test_df.drop(columns=["churn_flag", "est_annual_revenue"])
            y_test = test_df["churn_flag"]
            y_revenue_test = test_df["est_annual_revenue"]

            logging.info("Applying feature engineering")

            # Apply feature engineering to train and test sets
            X_train_enhanced = create_enhanced_features(X_train)
            X_test_enhanced = create_enhanced_features(X_test)

            logging.info(
                f"Enhanced features created. Train shape: {X_train_enhanced.shape}, Test shape: {X_test_enhanced.shape}"
            )

            # Identify categorical and numerical features
            cat_features = [
                "gender",
                "education_level",
                "marital_status",
                "income_category",
                "card_category",
            ]
            num_features = X_train_enhanced.columns[
                ~X_train_enhanced.columns.isin(cat_features)
            ].tolist()

            logging.info(f"Number of numerical features: {len(num_features)}")
            logging.info(f"Number of categorical features: {len(cat_features)}")

            # Create preprocessing objects
            logging.info("Creating preprocessing objects")
            preprocessor = self.get_data_transformer(num_features, cat_features)

            # Save enhanced data for classification
            train_enhanced_df = pd.concat(
                [X_train_enhanced, y_train, y_revenue_train], axis=1
            )
            test_enhanced_df = pd.concat(
                [X_test_enhanced, y_test, y_revenue_test], axis=1
            )

            os.makedirs(
                os.path.dirname(self.transformation_config.train_enhanced_data_path),
                exist_ok=True,
            )
            train_enhanced_df.to_csv(
                self.transformation_config.train_enhanced_data_path, index=False
            )
            test_enhanced_df.to_csv(
                self.transformation_config.test_enhanced_data_path, index=False
            )

            logging.info(
                f"Enhanced train data saved to {self.transformation_config.train_enhanced_data_path}"
            )
            logging.info(
                f"Enhanced test data saved to {self.transformation_config.test_enhanced_data_path}"
            )

            # Prepare data for clustering
            logging.info("Preparing data for clustering")

            # Select features for clustering
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

            # Create clustering dataset
            X_cluster = X_train_enhanced[clustering_features].copy()

            # Create and fit cluster preprocessor
            preprocessor_cluster = self.get_cluster_transformer(clustering_features)

            # Save cluster data
            os.makedirs(
                os.path.dirname(
                    self.transformation_config.train_enhanced_cluster_data_path
                ),
                exist_ok=True,
            )
            X_cluster.to_csv(
                self.transformation_config.train_enhanced_cluster_data_path, index=False
            )

            logging.info(
                f"Clustering data saved to {self.transformation_config.train_enhanced_cluster_data_path}"
            )

            # Prepare data for regression
            logging.info("Preparing data for regression")

            # Identify and remove features that would cause data leakage in revenue prediction
            leakage_columns = identify_leakage_columns([])

            # Create clean feature sets for revenue prediction
            X_train_revenue = X_train_enhanced.drop(columns=leakage_columns)
            X_test_revenue = X_test_enhanced.drop(columns=leakage_columns)

            # Update feature lists for regression
            cat_features_revenue = [
                col for col in cat_features if col in X_train_revenue.columns
            ]
            num_features_revenue = [
                col
                for col in X_train_revenue.columns
                if col not in cat_features_revenue
            ]

            # Create regression preprocessor
            preprocessor_reg = self.get_regression_transformer(
                num_features_revenue, cat_features_revenue
            )

            # Save regression data
            train_revenue_df = pd.concat([X_train_revenue, y_revenue_train], axis=1)
            test_revenue_df = pd.concat([X_test_revenue, y_revenue_test], axis=1)

            os.makedirs(
                os.path.dirname(
                    self.transformation_config.train_enhanced_reg_data_path
                ),
                exist_ok=True,
            )
            train_revenue_df.to_csv(
                self.transformation_config.train_enhanced_reg_data_path, index=False
            )
            test_revenue_df.to_csv(
                self.transformation_config.test_enhanced_reg_data_path, index=False
            )

            logging.info(
                f"Regression train data saved to {self.transformation_config.train_enhanced_reg_data_path}"
            )
            logging.info(
                f"Regression test data saved to {self.transformation_config.test_enhanced_reg_data_path}"
            )

            # Create PCA for visualization
            logging.info("Creating PCA for cluster visualization")
            pca = PCA(n_components=2)

            # Save preprocessor objects
            logging.info("Saving preprocessor objects")
            save_object(self.transformation_config.preprocessor_obj_path, preprocessor)
            save_object(
                self.transformation_config.preprocessor_cluster_obj_path,
                preprocessor_cluster,
            )
            save_object(
                self.transformation_config.preprocessor_reg_obj_path, preprocessor_reg
            )
            save_object(self.transformation_config.pca_obj_path, pca)

            logging.info("Data transformation completed successfully")

            # Return paths to transformed data and preprocessors
            return (
                self.transformation_config.train_enhanced_data_path,
                self.transformation_config.test_enhanced_data_path,
                self.transformation_config.train_enhanced_cluster_data_path,
                self.transformation_config.train_enhanced_reg_data_path,
                self.transformation_config.test_enhanced_reg_data_path,
                self.transformation_config.preprocessor_obj_path,
                self.transformation_config.preprocessor_cluster_obj_path,
                self.transformation_config.preprocessor_reg_obj_path,
                self.transformation_config.pca_obj_path,
            )

        except Exception as e:
            logging.error(f"Exception occurred during data transformation: {e}")
            raise CustomException(e, sys)
