import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import xgboost as xgb
import lightgbm as lgb

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    save_object,
    load_object,
    evaluate_classification_model,
    evaluate_regression_model,
    load_config,
)

from sklearn.model_selection import cross_val_score


@dataclass
class ModelTrainerConfig:
    """Configuration for model training process"""

    churn_model_path: str = os.path.join("artifacts", "models", "churn_model.pkl")
    revenue_model_path: str = os.path.join("artifacts", "models", "revenue_model.pkl")
    cluster_model_path: str = os.path.join("artifacts", "models", "cluster_model.pkl")
    churn_model_report_path: str = os.path.join(
        "artifacts", "reports", "churn_model_report.json"
    )
    revenue_model_report_path: str = os.path.join(
        "artifacts", "reports", "revenue_model_report.json"
    )
    cluster_model_report_path: str = os.path.join(
        "artifacts", "reports", "cluster_model_report.json"
    )
    config_path: str = os.path.join("config", "config.yaml")
    cluster_assignments_path: str = os.path.join(
        "artifacts", "data", "cluster_assignments.csv"
    )


class ModelTrainer:
    """Class responsible for training and tuning models"""

    def __init__(self, config_path=None):
        """
        Initialize ModelTrainer with configuration

        Args:
            config_path (str, optional): Path to configuration file
        """
        self.model_trainer_config = ModelTrainerConfig()

        # Override with custom config if provided
        if config_path:
            self.model_trainer_config.config_path = config_path
            custom_config = load_config(config_path)

            if "model_trainer" in custom_config:
                model_config = custom_config["model_trainer"]
                for key, value in model_config.items():
                    if hasattr(self.model_trainer_config, key):
                        setattr(self.model_trainer_config, key, value)

    def train_churn_model(self, train_data_path, preprocessor_path):
        """
        Train and tune classification model for churn prediction

        Args:
            train_data_path (str): Path to training data
            preprocessor_path (str): Path to preprocessor object

        Returns:
            str: Path to saved model
        """
        try:
            logging.info("Starting churn model training")

            # Load data and preprocessor
            train_df = pd.read_csv(train_data_path)
            preprocessor = load_object(preprocessor_path)

            # Split features and target
            X_train = train_df.drop(columns=["churn_flag", "est_annual_revenue"])
            y_train = train_df["churn_flag"]

            # Define classification models
            classification_models = {
                "Logistic Regression": LogisticRegression(
                    class_weight="balanced", max_iter=1000
                ),
                "Random Forest": RandomForestClassifier(
                    class_weight="balanced", n_estimators=100
                ),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
                "XGBoost": xgb.XGBClassifier(scale_pos_weight=5, n_estimators=100),
                "LightGBM": lgb.LGBMClassifier(
                    class_weight="balanced", n_estimators=100
                ),
                "SVM": SVC(class_weight="balanced", probability=True),
            }

            # Create model pipelines using preprocessor
            from sklearn.pipeline import Pipeline

            classification_pipelines = {}
            for name, model in classification_models.items():
                classification_pipelines[name] = Pipeline(
                    [("preprocessor", preprocessor), ("classifier", model)]
                )

            # Evaluate models with cross-validation
            logging.info("Evaluating baseline classification models")
            baseline_results = {}

            for name, model in classification_pipelines.items():
                # Cross-validation with ROC-AUC scoring
                cv_scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring="roc_auc",
                )
                baseline_results[name] = {
                    "mean_auc": cv_scores.mean(),
                    "std_auc": cv_scores.std(),
                    "cv_scores": cv_scores.tolist(),
                }
                logging.info(
                    f"{name} - Mean ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})"
                )

            # Find best model
            best_model_name = max(
                baseline_results, key=lambda k: baseline_results[k]["mean_auc"]
            )
            logging.info(
                f"Best performing model: {best_model_name} with mean ROC-AUC of {baseline_results[best_model_name]['mean_auc']:.4f}"
            )

            # Define hyperparameter search spaces
            param_grids = {
                "Logistic Regression": {
                    "classifier__C": [0.01, 0.1, 1.0, 10.0],
                    "classifier__penalty": ["l2"],
                },
                "Random Forest": {
                    "classifier__n_estimators": [100, 200],
                    "classifier__max_depth": [None, 10, 20],
                    "classifier__min_samples_split": [2, 5, 10],
                },
                "Gradient Boosting": {
                    "classifier__n_estimators": [100, 200],
                    "classifier__learning_rate": [0.01, 0.1, 0.2],
                    "classifier__max_depth": [3, 5, 7],
                },
                "XGBoost": {
                    "classifier__n_estimators": [100, 200],
                    "classifier__learning_rate": [0.01, 0.1, 0.2],
                    "classifier__max_depth": [3, 5, 7],
                    "classifier__colsample_bytree": [0.7, 0.8, 0.9],
                    "classifier__scale_pos_weight": [3, 5, 7],
                },
                "LightGBM": {
                    "classifier__n_estimators": [100, 200],
                    "classifier__learning_rate": [0.01, 0.1, 0.2],
                    "classifier__max_depth": [3, 5, 7],
                    "classifier__num_leaves": [31, 50, 70],
                },
                "SVM": {
                    "classifier__C": [0.1, 1, 10],
                    "classifier__gamma": ["scale", "auto"],
                    "classifier__kernel": ["rbf", "linear"],
                },
            }

            # Tune the best performing model
            best_pipeline = classification_pipelines[best_model_name]
            param_grid = param_grids[best_model_name]

            logging.info(f"Tuning hyperparameters for {best_model_name}")
            grid_search = GridSearchCV(
                best_pipeline,
                param_grid,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="roc_auc",
                n_jobs=-1,
                verbose=1,
            )

            grid_search.fit(X_train, y_train)

            logging.info(f"Best parameters: {grid_search.best_params_}")
            logging.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")

            # Get the optimized model
            optimized_classifier = grid_search.best_estimator_

            # Create directory and save model
            os.makedirs(
                os.path.dirname(self.model_trainer_config.churn_model_path),
                exist_ok=True,
            )
            save_object(
                self.model_trainer_config.churn_model_path, optimized_classifier
            )

            # Save model report
            os.makedirs(
                os.path.dirname(self.model_trainer_config.churn_model_report_path),
                exist_ok=True,
            )
            model_report = {
                "best_model": best_model_name,
                "baseline_results": baseline_results,
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
            }

            import json

            with open(self.model_trainer_config.churn_model_report_path, "w") as f:
                json.dump(model_report, f, indent=4)

            logging.info(
                f"Churn model saved to {self.model_trainer_config.churn_model_path}"
            )
            logging.info(
                f"Churn model report saved to {self.model_trainer_config.churn_model_report_path}"
            )

            return self.model_trainer_config.churn_model_path

        except Exception as e:
            logging.error(f"Exception occurred during churn model training: {e}")
            raise CustomException(e, sys)

    def train_revenue_model(self, train_data_path, preprocessor_path):
        """
        Train and tune regression model for revenue prediction

        Args:
            train_data_path (str): Path to training data
            preprocessor_path (str): Path to preprocessor object

        Returns:
            str: Path to saved model
        """
        try:
            logging.info("Starting revenue model training")

            # Load data and preprocessor
            train_df = pd.read_csv(train_data_path)
            preprocessor = load_object(preprocessor_path)

            # Split features and target
            X_train = train_df.drop(columns=["est_annual_revenue"])
            y_train = train_df["est_annual_revenue"]

            # Define regression models
            regression_models = {
                "Ridge": Ridge(alpha=1.0),
                "Lasso": Lasso(alpha=0.1),
                "Random Forest": RandomForestRegressor(n_estimators=100),
                "XGBoost": xgb.XGBRegressor(n_estimators=100),
                "LightGBM": lgb.LGBMRegressor(n_estimators=100),
            }

            # Create model pipelines using preprocessor
            from sklearn.pipeline import Pipeline

            regression_pipelines = {}
            for name, model in regression_models.items():
                regression_pipelines[name] = Pipeline(
                    [("preprocessor", preprocessor), ("regressor", model)]
                )

            # Evaluate models with cross-validation
            logging.info("Evaluating baseline regression models")
            baseline_results = {}

            for name, model in regression_pipelines.items():
                # Cross-validation with RMSE and R2 scoring
                rmse_scores = -cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=KFold(n_splits=5, shuffle=True, random_state=42),
                    scoring="neg_root_mean_squared_error",
                )

                r2_scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=KFold(n_splits=5, shuffle=True, random_state=42),
                    scoring="r2",
                )

                baseline_results[name] = {
                    "mean_rmse": rmse_scores.mean(),
                    "std_rmse": rmse_scores.std(),
                    "mean_r2": r2_scores.mean(),
                    "std_r2": r2_scores.std(),
                    "cv_rmse_scores": rmse_scores.tolist(),
                    "cv_r2_scores": r2_scores.tolist(),
                }

                logging.info(
                    f"{name} - Mean RMSE: ${rmse_scores.mean():.2f} (±${rmse_scores.std():.2f}) - Mean R²: {r2_scores.mean():.4f}"
                )

            # Find best model
            best_model_name = min(
                baseline_results, key=lambda k: baseline_results[k]["mean_rmse"]
            )
            logging.info(
                f"Best performing regression model: {best_model_name} with mean RMSE of ${baseline_results[best_model_name]['mean_rmse']:.2f}"
            )

            # Define hyperparameter search spaces
            param_grids_reg = {
                "Ridge": {
                    "regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                    "regressor__fit_intercept": [True, False],
                    "regressor__solver": [
                        "auto",
                        "svd",
                        "cholesky",
                        "lsqr",
                        "sparse_cg",
                        "sag",
                        "saga",
                    ],
                },
                "Lasso": {
                    "regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                    "regressor__max_iter": [1000, 2000, 3000],
                },
                "Random Forest": {
                    "regressor__n_estimators": [100, 200],
                    "regressor__max_depth": [None, 10, 20],
                    "regressor__min_samples_split": [2, 5, 10],
                    "regressor__min_samples_leaf": [1, 2, 4],
                },
                "XGBoost": {
                    "regressor__n_estimators": [100, 200],
                    "regressor__learning_rate": [0.01, 0.1, 0.2],
                    "regressor__max_depth": [3, 5, 7],
                    "regressor__colsample_bytree": [0.7, 0.8, 0.9],
                },
                "LightGBM": {
                    "regressor__n_estimators": [100, 200],
                    "regressor__learning_rate": [0.01, 0.1, 0.2],
                    "regressor__max_depth": [3, 5, 7],
                    "regressor__num_leaves": [31, 50, 70],
                },
            }

            # Tune the best performing model
            best_reg_pipeline = regression_pipelines[best_model_name]
            param_grid_reg = param_grids_reg[best_model_name]

            logging.info(
                f"Tuning hyperparameters for {best_model_name} regression model"
            )
            grid_search_reg = GridSearchCV(
                best_reg_pipeline,
                param_grid_reg,
                cv=KFold(n_splits=5, shuffle=True, random_state=42),
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                verbose=1,
            )

            grid_search_reg.fit(X_train, y_train)

            logging.info(f"Best parameters: {grid_search_reg.best_params_}")
            logging.info(
                f"Best cross-validation RMSE: ${-grid_search_reg.best_score_:.2f}"
            )

            # Get the optimized model
            optimized_regressor = grid_search_reg.best_estimator_

            # Create directory and save model
            os.makedirs(
                os.path.dirname(self.model_trainer_config.revenue_model_path),
                exist_ok=True,
            )
            save_object(
                self.model_trainer_config.revenue_model_path, optimized_regressor
            )

            # Save model report
            os.makedirs(
                os.path.dirname(self.model_trainer_config.revenue_model_report_path),
                exist_ok=True,
            )
            model_report = {
                "best_model": best_model_name,
                "baseline_results": baseline_results,
                "best_params": grid_search_reg.best_params_,
                "best_score": -grid_search_reg.best_score_,
            }

            import json

            with open(self.model_trainer_config.revenue_model_report_path, "w") as f:
                json.dump(model_report, f, indent=4)

            logging.info(
                f"Revenue model saved to {self.model_trainer_config.revenue_model_path}"
            )
            logging.info(
                f"Revenue model report saved to {self.model_trainer_config.revenue_model_report_path}"
            )

            return self.model_trainer_config.revenue_model_path

        except Exception as e:
            logging.error(f"Exception occurred during revenue model training: {e}")
            raise CustomException(e, sys)

    def train_cluster_model(self, cluster_data_path, preprocessor_cluster_path):
        """
        Train clustering model for customer segmentation

        Args:
            cluster_data_path (str): Path to data for clustering
            preprocessor_cluster_path (str): Path to cluster preprocessor object

        Returns:
            tuple: Path to saved model and cluster assignments
        """
        try:
            logging.info("Starting clustering model training")

            # Load data and preprocessor
            X_cluster = pd.read_csv(cluster_data_path)
            preprocessor_cluster = load_object(preprocessor_cluster_path)

            # Apply preprocessing to clustering data
            X_cluster_processed = preprocessor_cluster.fit_transform(X_cluster)

            logging.info(
                f"Clustering data shape after preprocessing: {X_cluster_processed.shape}"
            )

            # Find optimal number of clusters using silhouette score
            silhouette_scores = []
            inertia_values = []
            k_range = range(2, 11)

            for k in k_range:
                # Create and fit K-means model
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_cluster_processed)

                # Calculate silhouette score
                silhouette_avg = silhouette_score(X_cluster_processed, kmeans.labels_)
                silhouette_scores.append(silhouette_avg)

                # Calculate inertia (sum of squared distances)
                inertia_values.append(kmeans.inertia_)

                logging.info(
                    f"K={k}, Silhouette Score={silhouette_avg:.4f}, Inertia={kmeans.inertia_:.2f}"
                )

            # Determine optimal number of clusters based on silhouette score
            optimal_k = k_range[np.argmax(silhouette_scores)]
            logging.info(
                f"Optimal number of clusters based on silhouette score: {optimal_k}"
            )

            # Train final clustering model with optimal k
            kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            clusters = kmeans_final.fit_predict(X_cluster_processed)

            # Create directory and save model
            os.makedirs(
                os.path.dirname(self.model_trainer_config.cluster_model_path),
                exist_ok=True,
            )
            save_object(self.model_trainer_config.cluster_model_path, kmeans_final)

            # Save cluster assignments
            os.makedirs(
                os.path.dirname(self.model_trainer_config.cluster_assignments_path),
                exist_ok=True,
            )
            pd.DataFrame({"cluster": clusters}).to_csv(
                self.model_trainer_config.cluster_assignments_path, index=False
            )

            # Get cluster centers and transform back to original scale
            cluster_centers = preprocessor_cluster.inverse_transform(
                kmeans_final.cluster_centers_
            )
            cluster_centers_df = pd.DataFrame(
                cluster_centers, columns=X_cluster.columns
            )

            # Save model report
            os.makedirs(
                os.path.dirname(self.model_trainer_config.cluster_model_report_path),
                exist_ok=True,
            )
            model_report = {
                "optimal_k": optimal_k,
                "silhouette_scores": {
                    k: score for k, score in zip(k_range, silhouette_scores)
                },
                "inertia_values": {
                    k: inertia for k, inertia in zip(k_range, inertia_values)
                },
                "cluster_centers": cluster_centers_df.to_dict(),
                "cluster_distribution": pd.Series(clusters).value_counts().to_dict(),
            }

            import json

            with open(self.model_trainer_config.cluster_model_report_path, "w") as f:
                json.dump(model_report, f, indent=4)

            logging.info(
                f"Cluster model saved to {self.model_trainer_config.cluster_model_path}"
            )
            logging.info(
                f"Cluster assignments saved to {self.model_trainer_config.cluster_assignments_path}"
            )
            logging.info(
                f"Cluster model report saved to {self.model_trainer_config.cluster_model_report_path}"
            )

            return (
                self.model_trainer_config.cluster_model_path,
                self.model_trainer_config.cluster_assignments_path,
            )

        except Exception as e:
            logging.error(f"Exception occurred during cluster model training: {e}")
            raise CustomException(e, sys)
