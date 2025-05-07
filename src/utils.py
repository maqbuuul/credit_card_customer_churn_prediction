import os
import sys
import pandas as pd
import numpy as np
import pickle
import yaml
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
)

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle

    Args:
        file_path (str): Path where object will be saved
        obj: Object to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        logging.error(f"Error in save_object: {e}")
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from a file using pickle

    Args:
        file_path (str): Path where object is saved

    Returns:
        Object loaded from file
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logging.info(f"Object loaded successfully from: {file_path}")
        return obj

    except Exception as e:
        logging.error(f"Error in load_object: {e}")
        raise CustomException(e, sys)


def evaluate_classification_model(y_true, y_pred, y_prob=None):
    """
    Evaluate classification model performance

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for positive class

    Returns:
        dict: Dictionary of evaluation metrics
    """
    try:
        metrics = {}

        # Calculate standard classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred)
        metrics["recall"] = recall_score(y_true, y_pred)
        metrics["f1_score"] = f1_score(y_true, y_pred)

        # Calculate ROC-AUC if probabilities are provided
        if y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

        # Calculate confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

        # Generate classification report
        metrics["classification_report"] = classification_report(
            y_true, y_pred, output_dict=True
        )

        logging.info(f"Classification metrics calculated: {metrics}")
        return metrics

    except Exception as e:
        logging.error(f"Error in evaluate_classification_model: {e}")
        raise CustomException(e, sys)


def evaluate_regression_model(y_true, y_pred):
    """
    Evaluate regression model performance

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        dict: Dictionary of evaluation metrics
    """
    try:
        metrics = {}

        # Calculate regression metrics
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["r2_score"] = r2_score(y_true, y_pred)
        metrics["mean_abs_pct_error"] = (
            np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        )

        logging.info(f"Regression metrics calculated: {metrics}")
        return metrics

    except Exception as e:
        logging.error(f"Error in evaluate_regression_model: {e}")
        raise CustomException(e, sys)


def load_config(config_path):
    """
    Load YAML configuration file

    Args:
        config_path (str): Path to YAML configuration file

    Returns:
        dict: Configuration parameters
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        logging.info(f"Configuration loaded from: {config_path}")
        return config

    except Exception as e:
        logging.error(f"Error in load_config: {e}")
        raise CustomException(e, sys)


def save_config(config, config_path):
    """
    Save configuration to YAML file

    Args:
        config (dict): Configuration parameters to save
        config_path (str): Path where to save YAML file
    """
    try:
        dir_path = os.path.dirname(config_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(config_path, "w") as file:
            yaml.dump(config, file)

        logging.info(f"Configuration saved to: {config_path}")

    except Exception as e:
        logging.error(f"Error in save_config: {e}")
        raise CustomException(e, sys)


def create_enhanced_features(df):
    """
    Generate enhanced features for credit card churn analysis

    Args:
        df (pd.DataFrame): Input DataFrame with original features

    Returns:
        pd.DataFrame: DataFrame with added enhanced features
    """
    try:
        # Create a copy to avoid modifying the original
        enhanced_df = df.copy()

        # Ratio features
        enhanced_df["trans_per_dollar"] = enhanced_df["total_trans_ct"] / (
            enhanced_df["credit_limit"] + 1
        )
        enhanced_df["avg_transaction_value"] = enhanced_df["total_trans_amt"] / (
            enhanced_df["total_trans_ct"] + 1
        )
        enhanced_df["utilization_ratio"] = enhanced_df["total_revolving_bal"] / (
            enhanced_df["credit_limit"] + 1
        )
        enhanced_df["inactive_vs_tenure_ratio"] = enhanced_df[
            "months_inactive_12_mon"
        ] / (enhanced_df["months_on_book"] + 1)
        enhanced_df["contacts_per_month"] = (
            12
            * enhanced_df["contacts_count_12_mon"]
            / (enhanced_df["months_on_book"] + 1)
        )
        enhanced_df["product_penetration"] = (
            enhanced_df["total_relationship_count"] / 6
        )  # 6 is max possible products

        # Change indicators
        enhanced_df["trans_count_change_ratio"] = enhanced_df["total_ct_chng_q4_q1"]
        enhanced_df["trans_amount_change_ratio"] = enhanced_df["total_amt_chng_q4_q1"]
        enhanced_df["is_active"] = (enhanced_df["months_inactive_12_mon"] == 0).astype(
            int
        )
        enhanced_df["has_zero_balance"] = (
            enhanced_df["total_revolving_bal"] == 0
        ).astype(int)

        # Interaction features
        enhanced_df["inactive_contacts"] = (
            enhanced_df["months_inactive_12_mon"] * enhanced_df["contacts_count_12_mon"]
        )
        enhanced_df["utilization_contacts"] = (
            enhanced_df["avg_utilization_ratio"] * enhanced_df["contacts_count_12_mon"]
        )
        enhanced_df["tenure_product_ratio"] = enhanced_df["months_on_book"] / (
            enhanced_df["total_relationship_count"] + 1
        )

        # Customer value features
        enhanced_df["avg_spend_per_month"] = enhanced_df["total_trans_amt"] / 12
        enhanced_df["revolving_to_trans_ratio"] = enhanced_df["total_revolving_bal"] / (
            enhanced_df["total_trans_amt"] + 1
        )

        logging.info(f"Enhanced features created, new shape: {enhanced_df.shape}")
        return enhanced_df

    except Exception as e:
        logging.error(f"Error in create_enhanced_features: {e}")
        raise CustomException(e, sys)


def identify_leakage_columns(target_related_features):
    """
    Identify columns that would cause data leakage based on predefined list

    Args:
        target_related_features (list): Features related to the target

    Returns:
        list: Columns identified as potential leakage sources
    """
    try:
        # Common patterns for leakage features
        leakage_patterns = [
            "total_revolving_bal",  # Directly used in revenue calculation
            "total_trans_amt",  # Directly used in revenue calculation
            "avg_utilization_ratio",  # Derived from revolving balance
            "revolving_to_trans_ratio",  # Derived from target-related variables
            "avg_spend_per_month",  # Derived from transaction amount
            "has_zero_balance",  # Derived from revolving balance
            "trans_per_dollar",  # Contains transaction amount information
        ]

        # Add user-specified target-related features
        leakage_columns = leakage_patterns + target_related_features

        logging.info(f"Identified potential leakage columns: {leakage_columns}")
        return leakage_columns

    except Exception as e:
        logging.error(f"Error in identify_leakage_columns: {e}")
        raise CustomException(e, sys)
