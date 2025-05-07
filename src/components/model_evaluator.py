import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json
from sklearn.model_selection import cross_val_score

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    load_object,
    evaluate_classification_model,
    evaluate_regression_model,
    load_config,
)


@dataclass
class ModelEvaluatorConfig:
    """Configuration for model evaluation process"""

    churn_evaluation_path: str = os.path.join(
        "artifacts", "evaluations", "churn_evaluation.json"
    )
    revenue_evaluation_path: str = os.path.join(
        "artifacts", "evaluations", "revenue_evaluation.json"
    )
    integrated_evaluation_path: str = os.path.join(
        "artifacts", "evaluations", "integrated_evaluation.json"
    )
    revenue_at_risk_path: str = os.path.join("artifacts", "data", "revenue_at_risk.csv")
    config_path: str = os.path.join("config", "config.yaml")


class ModelEvaluator:
    """Class responsible for evaluating models on test data"""

    def __init__(self, config_path=None):
        """
        Initialize ModelEvaluator with configuration

        Args:
            config_path (str, optional): Path to configuration file
        """
        self.evaluator_config = ModelEvaluatorConfig()

        # Override with custom config if provided
        if config_path:
            self.evaluator_config.config_path = config_path
            custom_config = load_config(config_path)

            if "model_evaluator" in custom_config:
                eval_config = custom_config["model_evaluator"]
                for key, value in eval_config.items():
                    if hasattr(self.evaluator_config, key):
                        setattr(self.evaluator_config, key, value)

    def evaluate_churn_model(self, test_data_path, model_path):
        """
        Evaluate churn prediction model on test data

        Args:
            test_data_path (str): Path to test data
            model_path (str): Path to trained model

        Returns:
            dict: Evaluation metrics
        """
        try:
            logging.info("Starting churn model evaluation")

            # Load test data and model
            test_df = pd.read_csv(test_data_path)
            churn_model = load_object(model_path)

            # Split features and target
            X_test = test_df.drop(columns=["churn_flag", "est_annual_revenue"])
            y_test = test_df["churn_flag"]

            # Generate predictions
            y_pred_proba = churn_model.predict_proba(X_test)[:, 1]
            y_pred = churn_model.predict(X_test)

            # Calculate performance metrics
            metrics = evaluate_classification_model(y_test, y_pred, y_pred_proba)

            # Save evaluation results
            os.makedirs(
                os.path.dirname(self.evaluator_config.churn_evaluation_path),
                exist_ok=True,
            )

            # Convert numpy arrays to lists for JSON serialization
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics[key] = value.tolist()

            with open(self.evaluator_config.churn_evaluation_path, "w") as f:
                json.dump(metrics, f, indent=4)

            logging.info(
                f"Churn model evaluation results saved to {self.evaluator_config.churn_evaluation_path}"
            )
            logging.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logging.info(f"Precision: {metrics['precision']:.4f}")
            logging.info(f"Recall: {metrics['recall']:.4f}")
            logging.info(f"F1 Score: {metrics['f1_score']:.4f}")

            return metrics

        except Exception as e:
            logging.error(f"Exception occurred during churn model evaluation: {e}")
            raise CustomException(e, sys)

    def evaluate_revenue_model(self, test_data_path, model_path):
        """
        Evaluate revenue prediction model on test data

        Args:
            test_data_path (str): Path to test data
            model_path (str): Path to trained model

        Returns:
            dict: Evaluation metrics
        """
        try:
            logging.info("Starting revenue model evaluation")

            # Load test data and model
            test_df = pd.read_csv(test_data_path)
            revenue_model = load_object(model_path)

            # Split features and target
            X_test = test_df.drop(columns=["est_annual_revenue"])
            y_test = test_df["est_annual_revenue"]

            # Generate predictions
            y_pred = revenue_model.predict(X_test)

            # Calculate performance metrics
            metrics = evaluate_regression_model(y_test, y_pred)

            # Save evaluation results
            os.makedirs(
                os.path.dirname(self.evaluator_config.revenue_evaluation_path),
                exist_ok=True,
            )
            with open(self.evaluator_config.revenue_evaluation_path, "w") as f:
                json.dump(metrics, f, indent=4)

            logging.info(
                f"Revenue model evaluation results saved to {self.evaluator_config.revenue_evaluation_path}"
            )
            logging.info(f"RMSE: ${metrics['rmse']:.2f}")
            logging.info(f"R²: {metrics['r2_score']:.4f}")
            logging.info(
                f"Mean Absolute Percentage Error: {metrics['mean_abs_pct_error']:.2f}%"
            )

            return metrics

        except Exception as e:
            logging.error(f"Exception occurred during revenue model evaluation: {e}")
            raise CustomException(e, sys)

    def evaluate_integrated_model(
        self, test_data_path, churn_model_path, revenue_model_path
    ):
        """
        Evaluate integrated model (churn + revenue) to calculate revenue at risk

        Args:
            test_data_path (str): Path to test data
            churn_model_path (str): Path to churn prediction model
            revenue_model_path (str): Path to revenue prediction model

        Returns:
            dict: Integrated evaluation metrics
        """
        try:
            logging.info("Starting integrated model evaluation")

            # Load test data and models
            test_df = pd.read_csv(test_data_path)
            churn_model = load_object(churn_model_path)
            revenue_model = load_object(revenue_model_path)

            # Split features and targets
            X_test = test_df.drop(columns=["churn_flag", "est_annual_revenue"])
            y_test_churn = test_df["churn_flag"]
            y_test_revenue = test_df["est_annual_revenue"]

            # Generate predictions
            churn_proba = churn_model.predict_proba(X_test)[:, 1]
            revenue_pred = revenue_model.predict(X_test)

            # Calculate revenue at risk
            revenue_at_risk = churn_proba * revenue_pred

            # Create a DataFrame with results
            results_df = pd.DataFrame(
                {
                    "churn_probability": churn_proba,
                    "estimated_annual_revenue": revenue_pred,
                    "revenue_at_risk": revenue_at_risk,
                    "actual_churn": y_test_churn.values,
                    "actual_revenue": y_test_revenue.values,
                }
            )

            # Calculate high-risk customers (probability > 0.5)
            high_risk_mask = results_df["churn_probability"] > 0.5
            results_df["risk_category"] = np.where(
                high_risk_mask, "High Risk", "Low Risk"
            )

            # Define value segments
            revenue_bins = pd.qcut(
                results_df["estimated_annual_revenue"],
                4,
                labels=[
                    "Low Value",
                    "Medium-Low Value",
                    "Medium-High Value",
                    "High Value",
                ],
            )
            results_df["value_segment"] = revenue_bins

            # Calculate integrated metrics
            total_revenue_at_risk = results_df["revenue_at_risk"].sum()
            avg_revenue_at_risk = results_df["revenue_at_risk"].mean()
            pct_high_risk = 100 * high_risk_mask.mean()

            # Calculate risk matrix by revenue
            risk_matrix = pd.crosstab(
                results_df["value_segment"],
                results_df["risk_category"],
                values=results_df["revenue_at_risk"],
                aggfunc="sum",
                normalize=False,
            ).to_dict()

            # Calculate risk matrix by customer count
            risk_matrix_count = pd.crosstab(
                results_df["value_segment"],
                results_df["risk_category"],
                normalize=False,
            ).to_dict()

            # Create integrated metrics
            integrated_metrics = {
                "total_revenue_at_risk": float(total_revenue_at_risk),
                "avg_revenue_at_risk": float(avg_revenue_at_risk),
                "pct_high_risk": float(pct_high_risk),
                "risk_matrix_revenue": risk_matrix,
                "risk_matrix_count": risk_matrix_count,
            }

            # Save evaluation results
            os.makedirs(
                os.path.dirname(self.evaluator_config.integrated_evaluation_path),
                exist_ok=True,
            )
            with open(self.evaluator_config.integrated_evaluation_path, "w") as f:
                json.dump(integrated_metrics, f, indent=4)

            # Save revenue at risk DataFrame
            os.makedirs(
                os.path.dirname(self.evaluator_config.revenue_at_risk_path),
                exist_ok=True,
            )
            results_df.to_csv(self.evaluator_config.revenue_at_risk_path, index=False)

            logging.info(
                f"Integrated model evaluation saved to {self.evaluator_config.integrated_evaluation_path}"
            )
            logging.info(
                f"Revenue at risk data saved to {self.evaluator_config.revenue_at_risk_path}"
            )
            logging.info(f"Total annual revenue at risk: ${total_revenue_at_risk:,.2f}")
            logging.info(
                f"Average revenue at risk per customer: ${avg_revenue_at_risk:,.2f}"
            )
            logging.info(f"Percentage of high-risk customers: {pct_high_risk:.2f}%")

            return integrated_metrics

        except Exception as e:
            logging.error(f"Exception occurred during integrated model evaluation: {e}")
            raise CustomException(e, sys)
