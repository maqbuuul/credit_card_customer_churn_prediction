import os
import sys
import pandas as pd
import yaml
from datetime import datetime

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluator import ModelEvaluator
from src.components.visualizer import Visualizer
from src.utils import save_config


class TrainingPipeline:
    """
    End-to-end training pipeline for credit card churn analysis

    This pipeline orchestrates the complete modeling process:
    1. Data ingestion
    2. Data transformation and feature engineering
    3. Model training (churn, revenue, clustering)
    4. Model evaluation
    5. Visualization
    """

    def __init__(self, config_path=None):
        """
        Initialize the training pipeline

        Args:
            config_path (str, optional): Path to configuration file
        """
        self.config_path = config_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create default config if none provided
        if not config_path:
            self.config_path = self._create_default_config()

    def _create_default_config(self):
        """
        Create default configuration file

        Returns:
            str: Path to created configuration file
        """
        try:
            # Default configuration
            default_config = {
                "timestamp": self.timestamp,
                "data_paths": {
                    "data_path": "data/processed/clean_bank_data.csv",  # FIXED: Removed leading ../
                },
                "data_ingestion": {
                    "raw_data_path": os.path.join("artifacts", "data", "raw.csv"),
                    "train_data_path": os.path.join("artifacts", "data", "train.csv"),
                    "test_data_path": os.path.join("artifacts", "data", "test.csv"),
                },
                "data_transformation": {
                    "preprocessor_obj_path": os.path.join(
                        "artifacts", "preprocessor", "preprocessor.pkl"
                    ),
                    "preprocessor_cluster_obj_path": os.path.join(
                        "artifacts", "preprocessor", "cluster_preprocessor.pkl"
                    ),
                    "preprocessor_reg_obj_path": os.path.join(
                        "artifacts", "preprocessor", "reg_preprocessor.pkl"
                    ),
                    "pca_obj_path": os.path.join(
                        "artifacts", "preprocessor", "pca.pkl"
                    ),
                },
                "model_trainer": {
                    "churn_model_path": os.path.join(
                        "artifacts", "models", "churn_model.pkl"
                    ),
                    "revenue_model_path": os.path.join(
                        "artifacts", "models", "revenue_model.pkl"
                    ),
                    "cluster_model_path": os.path.join(
                        "artifacts", "models", "cluster_model.pkl"
                    ),
                },
                "model_evaluator": {
                    "churn_evaluation_path": os.path.join(
                        "artifacts", "evaluations", "churn_evaluation.json"
                    ),
                    "revenue_evaluation_path": os.path.join(
                        "artifacts", "evaluations", "revenue_evaluation.json"
                    ),
                    "integrated_evaluation_path": os.path.join(
                        "artifacts", "evaluations", "integrated_evaluation.json"
                    ),
                    "revenue_at_risk_path": os.path.join(
                        "artifacts", "data", "revenue_at_risk.csv"
                    ),
                },
                "visualizer": {
                    "output_dir": os.path.join("artifacts", "visualizations")
                },
            }

            # Create config directory if it doesn't exist
            os.makedirs("config", exist_ok=True)

            # Create timestamped config file
            config_path = os.path.join("config", f"config_{self.timestamp}.yaml")

            # Save configuration
            with open(config_path, "w") as f:
                yaml.dump(default_config, f, default_flow_style=False)

            logging.info(f"Created default configuration at {config_path}")
            return config_path

        except Exception as e:
            logging.error(f"Error creating default configuration: {e}")
            raise CustomException(e, sys)

    def run(self, data_path=None):
        """
        Execute the complete training pipeline

        Args:
            data_path (str, optional): Path to input data

        Returns:
            dict: Summary of training results
        """
        try:
            logging.info(f"Starting training pipeline with config: {self.config_path}")

            # 1. Data Ingestion
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(config_path=self.config_path)
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(
                data_path
            )

            # 2. Data Transformation
            logging.info("Starting data transformation")
            data_transformation = DataTransformation(config_path=self.config_path)
            (
                train_enhanced_path,
                test_enhanced_path,
                train_cluster_path,
                train_revenue_path,
                test_revenue_path,
                preprocessor_path,
                preprocessor_cluster_path,
                preprocessor_reg_path,
                pca_path,
            ) = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )

            # 3. Model Training
            logging.info("Starting model training")
            model_trainer = ModelTrainer(config_path=self.config_path)

            # 3.1 Train Churn Model
            logging.info("Training churn prediction model")
            churn_model_path = model_trainer.train_churn_model(
                train_enhanced_path, preprocessor_path
            )

            # 3.2 Train Revenue Model
            logging.info("Training revenue prediction model")
            revenue_model_path = model_trainer.train_revenue_model(
                train_revenue_path, preprocessor_reg_path
            )

            # 3.3 Train Clustering Model
            logging.info("Training customer segmentation model")
            cluster_model_path, cluster_assignments_path = (
                model_trainer.train_cluster_model(
                    train_cluster_path, preprocessor_cluster_path
                )
            )

            # 4. Model Evaluation
            logging.info("Starting model evaluation")
            model_evaluator = ModelEvaluator(config_path=self.config_path)

            # 4.1 Evaluate Churn Model
            logging.info("Evaluating churn model")
            churn_metrics = model_evaluator.evaluate_churn_model(
                test_enhanced_path, churn_model_path
            )

            # 4.2 Evaluate Revenue Model
            logging.info("Evaluating revenue model")
            revenue_metrics = model_evaluator.evaluate_revenue_model(
                test_revenue_path, revenue_model_path
            )

            # 4.3 Evaluate Integrated Model
            logging.info("Evaluating integrated model")
            integrated_metrics = model_evaluator.evaluate_integrated_model(
                test_enhanced_path, churn_model_path, revenue_model_path
            )

            # 5. Visualization
            logging.info("Generating visualizations")
            visualizer = Visualizer(config_path=self.config_path)

            # 5.1 Churn Model Visualizations
            churn_viz_paths = visualizer.generate_churn_model_visuals(
                test_enhanced_path,
                churn_model_path,
                model_evaluator.evaluator_config.churn_evaluation_path,
            )

            # 5.2 Revenue Model Visualizations
            revenue_viz_paths = visualizer.generate_revenue_model_visuals(
                test_revenue_path,
                revenue_model_path,
                model_evaluator.evaluator_config.revenue_evaluation_path,
            )

            # 5.3 Cluster Visualizations
            cluster_viz_paths = visualizer.generate_cluster_visuals(
                train_cluster_path,
                preprocessor_cluster_path,
                cluster_model_path,
                cluster_assignments_path,
                pca_path,
            )

            # 5.4 Integrated Model Visualizations
            integrated_viz_paths = visualizer.generate_integrated_visuals(
                model_evaluator.evaluator_config.revenue_at_risk_path,
                model_evaluator.evaluator_config.integrated_evaluation_path,
            )

            # 6. Create Results Summary
            logging.info("Creating results summary")
            results_summary = {
                "timestamp": self.timestamp,
                "config_path": self.config_path,
                "churn_model": {
                    "model_path": churn_model_path,
                    "metrics": churn_metrics,
                    "visualizations": churn_viz_paths,
                },
                "revenue_model": {
                    "model_path": revenue_model_path,
                    "metrics": revenue_metrics,
                    "visualizations": revenue_viz_paths,
                },
                "cluster_model": {
                    "model_path": cluster_model_path,
                    "visualizations": cluster_viz_paths,
                },
                "integrated_model": {
                    "metrics": integrated_metrics,
                    "visualizations": integrated_viz_paths,
                },
            }

            # Save summary to file
            summary_path = os.path.join(
                "artifacts", "summary", f"training_summary_{self.timestamp}.json"
            )
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)

            import json

            with open(summary_path, "w") as f:
                json.dump(results_summary, f, indent=4)

            logging.info(
                f"Training pipeline completed successfully. Summary saved to {summary_path}"
            )

            return results_summary

        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise CustomException(e, sys)
