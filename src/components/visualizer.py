import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import json
from math import pi

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, load_config


@dataclass
class VisualizerConfig:
    """Configuration for visualization process"""

    output_dir: str = os.path.join("artifacts", "visualizations")
    churn_viz_dir: str = os.path.join("artifacts", "visualizations", "churn")
    revenue_viz_dir: str = os.path.join("artifacts", "visualizations", "revenue")
    cluster_viz_dir: str = os.path.join("artifacts", "visualizations", "cluster")
    integrated_viz_dir: str = os.path.join("artifacts", "visualizations", "integrated")
    config_path: str = os.path.join("config", "config.yaml")


class Visualizer:
    """Class responsible for creating visualizations of model results"""

    def __init__(self, config_path=None):
        """
        Initialize Visualizer with configuration

        Args:
            config_path (str, optional): Path to configuration file
        """
        self.visualizer_config = VisualizerConfig()

        # Set custom color palette for consistent branding
        self.color_palette = [
            "#023047",
            "#e85d04",
            "#0077b6",
            "#ff8200",
            "#0096c7",
            "#ff9c33",
        ]
        sns.set_palette(sns.color_palette(self.color_palette))
        sns.set_theme(style="ticks")

        # Override with custom config if provided
        if config_path:
            self.visualizer_config.config_path = config_path
            custom_config = load_config(config_path)

            if "visualizer" in custom_config:
                viz_config = custom_config["visualizer"]
                for key, value in viz_config.items():
                    if hasattr(self.visualizer_config, key):
                        setattr(self.visualizer_config, key, value)

        # Create visualization directories
        os.makedirs(self.visualizer_config.churn_viz_dir, exist_ok=True)
        os.makedirs(self.visualizer_config.revenue_viz_dir, exist_ok=True)
        os.makedirs(self.visualizer_config.cluster_viz_dir, exist_ok=True)
        os.makedirs(self.visualizer_config.integrated_viz_dir, exist_ok=True)

    def generate_churn_model_visuals(self, test_data_path, model_path, evaluation_path):
        """
        Generate visualizations for churn model performance

        Args:
            test_data_path (str): Path to test data
            model_path (str): Path to trained model
            evaluation_path (str): Path to model evaluation results

        Returns:
            list: Paths to generated visualizations
        """
        try:
            logging.info("Generating churn model visualizations")

            # Load test data, model, and evaluation results
            test_df = pd.read_csv(test_data_path)
            churn_model = load_object(model_path)

            with open(evaluation_path, "r") as f:
                evaluation = json.load(f)

            # Split features and target
            X_test = test_df.drop(columns=["churn_flag", "est_annual_revenue"])
            y_test = test_df["churn_flag"]

            # Generate predictions
            y_pred_proba = churn_model.predict_proba(X_test)[:, 1]
            y_pred = churn_model.predict(X_test)

            generated_visuals = []

            # Plot 1: Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = np.array(evaluation["confusion_matrix"])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"],
            )
            plt.xlabel("Predicted", fontsize=12)
            plt.ylabel("Actual", fontsize=12)
            plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
            plt.tight_layout()

            confusion_matrix_path = os.path.join(
                self.visualizer_config.churn_viz_dir, "confusion_matrix.png"
            )
            plt.savefig(confusion_matrix_path)
            plt.close()
            generated_visuals.append(confusion_matrix_path)

            # Plot 2: ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(
                fpr,
                tpr,
                color="blue",
                lw=2,
                label=f"ROC curve (AUC = {evaluation['roc_auc']:.4f})",
            )
            plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)
            plt.title(
                "Receiver Operating Characteristic (ROC) Curve",
                fontsize=14,
                fontweight="bold",
            )
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            roc_curve_path = os.path.join(
                self.visualizer_config.churn_viz_dir, "roc_curve.png"
            )
            plt.savefig(roc_curve_path)
            plt.close()
            generated_visuals.append(roc_curve_path)

            # Plot 3: Precision-Recall Curve
            precision_curve, recall_curve, _ = precision_recall_curve(
                y_test, y_pred_proba
            )
            plt.figure(figsize=(8, 6))
            plt.plot(recall_curve, precision_curve, color="green", lw=2)
            plt.xlabel("Recall", fontsize=12)
            plt.ylabel("Precision", fontsize=12)
            plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            pr_curve_path = os.path.join(
                self.visualizer_config.churn_viz_dir, "precision_recall_curve.png"
            )
            plt.savefig(pr_curve_path)
            plt.close()
            generated_visuals.append(pr_curve_path)

            # Plot 4: Feature importance (if available)
            if hasattr(churn_model.named_steps["classifier"], "feature_importances_"):
                # Get feature names after preprocessing (simplified approach)
                feature_names = list(X_test.columns)

                # Get feature importances
                feature_importances = churn_model.named_steps[
                    "classifier"
                ].feature_importances_

                # Sort features by importance, keeping track of original indices
                sorted_indices = np.argsort(feature_importances)[::-1]

                # Plot feature importances using original feature names
                plt.figure(figsize=(12, 8))
                plt.title(
                    "Feature Importance for Churn Prediction",
                    fontsize=14,
                    fontweight="bold",
                )

                # Use feature_names for x-axis labels, limited to top 20
                plt.bar(
                    range(min(20, len(sorted_indices))),
                    feature_importances[sorted_indices[:20]],
                    color="#0077b6",
                )
                plt.xticks(
                    range(min(20, len(sorted_indices))),
                    [feature_names[i] for i in sorted_indices[:20]],
                    rotation=90,
                )

                plt.xlabel("Features", fontsize=12)
                plt.ylabel("Importance", fontsize=12)
                plt.tight_layout()

                feature_importance_path = os.path.join(
                    self.visualizer_config.churn_viz_dir, "feature_importance.png"
                )
                plt.savefig(feature_importance_path)
                plt.close()
                generated_visuals.append(feature_importance_path)

            logging.info(
                f"Generated {len(generated_visuals)} churn model visualizations"
            )
            return generated_visuals

        except Exception as e:
            logging.error(f"Exception occurred during churn visualization: {e}")
            raise CustomException(e, sys)

    def generate_revenue_model_visuals(
        self, test_data_path, model_path, evaluation_path
    ):
        """
        Generate visualizations for revenue model performance

        Args:
            test_data_path (str): Path to test data
            model_path (str): Path to trained model
            evaluation_path (str): Path to model evaluation results

        Returns:
            list: Paths to generated visualizations
        """
        try:
            logging.info("Generating revenue model visualizations")

            # Load test data, model, and evaluation results
            test_df = pd.read_csv(test_data_path)
            revenue_model = load_object(model_path)

            with open(evaluation_path, "r") as f:
                evaluation = json.load(f)

            # Split features and target
            X_test = test_df.drop(columns=["est_annual_revenue"])
            y_test = test_df["est_annual_revenue"]

            # Generate predictions
            y_pred = revenue_model.predict(X_test)

            generated_visuals = []

            # Plot 1: Actual vs Predicted Values
            plt.figure(figsize=(10, 8))
            sns.regplot(
                x=y_test,
                y=y_pred,
                scatter_kws={"alpha": 0.5, "color": "#0077b6"},
                line_kws={"color": "k", "linestyle": "--"},
            )
            plt.xlabel("Actual Annual Revenue ($)", fontsize=12)
            plt.ylabel("Predicted Annual Revenue ($)", fontsize=12)
            plt.title(
                "Actual vs. Predicted Annual Revenue", fontsize=14, fontweight="bold"
            )
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            actual_vs_pred_path = os.path.join(
                self.visualizer_config.revenue_viz_dir, "actual_vs_predicted.png"
            )
            plt.savefig(actual_vs_pred_path)
            plt.close()
            generated_visuals.append(actual_vs_pred_path)

            # Plot 2: Residuals
            residuals = y_test - y_pred
            plt.figure(figsize=(10, 6))
            sns.residplot(
                x=y_pred, y=residuals, scatter_kws={"alpha": 0.5, "color": "#e85d04"}
            )
            plt.axhline(y=0, color="k", linestyle="--", lw=2)
            plt.xlabel("Predicted Annual Revenue ($)", fontsize=12)
            plt.ylabel("Residuals ($)", fontsize=12)
            plt.title("Residual Plot", fontsize=14, fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            residuals_path = os.path.join(
                self.visualizer_config.revenue_viz_dir, "residuals.png"
            )
            plt.savefig(residuals_path)
            plt.close()
            generated_visuals.append(residuals_path)

            # Plot 3: Residuals Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(residuals, bins=30, color="#0096c7")
            plt.xlabel("Residuals ($)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.title("Residuals Distribution", fontsize=14, fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            residuals_dist_path = os.path.join(
                self.visualizer_config.revenue_viz_dir, "residuals_distribution.png"
            )
            plt.savefig(residuals_dist_path)
            plt.close()
            generated_visuals.append(residuals_dist_path)

            # Plot 4: Feature importance (if available)
            if hasattr(revenue_model.named_steps["regressor"], "feature_importances_"):
                # Get feature names
                feature_names = list(X_test.columns)

                # Get feature importances
                feature_importances = revenue_model.named_steps[
                    "regressor"
                ].feature_importances_

                # Create a DataFrame for feature importances
                feature_importance_df = pd.DataFrame(
                    {"Feature": feature_names, "Importance": feature_importances}
                )

                # Sort by importance
                feature_importance_df = feature_importance_df.sort_values(
                    by="Importance", ascending=False
                )

                # Plot feature importances for revenue prediction
                plt.figure(figsize=(12, 8))
                sns.barplot(
                    data=feature_importance_df.head(20),
                    x="Importance",
                    y="Feature",
                    color="#0096c7",
                )
                plt.title(
                    "Feature Importance for Revenue Prediction",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.xlabel("Importance", fontsize=12)
                plt.ylabel("Features", fontsize=12)
                plt.tight_layout()

                feature_importance_path = os.path.join(
                    self.visualizer_config.revenue_viz_dir, "feature_importance.png"
                )
                plt.savefig(feature_importance_path)
                plt.close()
                generated_visuals.append(feature_importance_path)

            logging.info(
                f"Generated {len(generated_visuals)} revenue model visualizations"
            )
            return generated_visuals

        except Exception as e:
            logging.error(f"Exception occurred during revenue visualization: {e}")
            raise CustomException(e, sys)

    def generate_cluster_visuals(
        self,
        cluster_data_path,
        preprocessor_path,
        model_path,
        cluster_assignments_path,
        pca_path,
    ):
        """
        Generate visualizations for customer segmentation

        Args:
            cluster_data_path (str): Path to clustering data
            preprocessor_path (str): Path to cluster preprocessor
            model_path (str): Path to clustering model
            cluster_assignments_path (str): Path to cluster assignments
            pca_path (str): Path to PCA object

        Returns:
            list: Paths to generated visualizations
        """
        try:
            logging.info("Generating clustering visualizations")

            # Load data and models
            X_cluster = pd.read_csv(cluster_data_path)
            preprocessor_cluster = load_object(preprocessor_path)
            kmeans_model = load_object(model_path)
            cluster_assignments = pd.read_csv(cluster_assignments_path)
            pca = load_object(pca_path)

            # Apply preprocessing to clustering data
            X_cluster_processed = preprocessor_cluster.transform(X_cluster)

            # Get cluster assignments
            clusters = cluster_assignments["cluster"].values

            # Get cluster centers
            cluster_centers = kmeans_model.cluster_centers_

            generated_visuals = []

            # Plot 1: PCA visualization of clusters
            X_pca = pca.fit_transform(X_cluster_processed)

            # Add PCA components and cluster to DataFrame
            X_pca_df = pd.DataFrame(
                {"PCA1": X_pca[:, 0], "PCA2": X_pca[:, 1], "Cluster": clusters}
            )

            # Plot clusters with PCA
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                data=X_pca_df,
                x="PCA1",
                y="PCA2",
                hue="Cluster",
                palette="viridis",
                s=100,
                alpha=0.7,
            )

            # Add cluster centers to the plot
            centers_pca = pca.transform(cluster_centers)
            plt.scatter(
                centers_pca[:, 0],
                centers_pca[:, 1],
                s=200,
                marker="X",
                c="red",
                alpha=0.9,
            )

            plt.title(
                "Customer Segments Visualization (PCA)", fontsize=14, fontweight="bold"
            )
            plt.xlabel("Principal Component 1", fontsize=12)
            plt.ylabel("Principal Component 2", fontsize=12)
            plt.legend(
                title="Cluster",
                title_fontsize=12,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            pca_path = os.path.join(
                self.visualizer_config.cluster_viz_dir, "cluster_pca.png"
            )
            plt.savefig(pca_path)
            plt.close()
            generated_visuals.append(pca_path)

            # Plot 2: Cluster Distribution
            plt.figure(figsize=(10, 6))
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            cluster_counts.plot(kind="bar", color=self.color_palette)
            plt.xlabel("Cluster", fontsize=12)
            plt.ylabel("Number of Customers", fontsize=12)
            plt.title(
                "Customer Distribution by Cluster", fontsize=14, fontweight="bold"
            )
            plt.xticks(rotation=0)
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()

            distribution_path = os.path.join(
                self.visualizer_config.cluster_viz_dir, "cluster_distribution.png"
            )
            plt.savefig(distribution_path)
            plt.close()
            generated_visuals.append(distribution_path)

            # Plot 3: Radar charts for each cluster
            # Note: This is a simplified version (would need cluster statistics)

            logging.info(f"Generated {len(generated_visuals)} cluster visualizations")
            return generated_visuals

        except Exception as e:
            logging.error(f"Exception occurred during cluster visualization: {e}")
            raise CustomException(e, sys)

    def generate_integrated_visuals(self, revenue_at_risk_path, integrated_eval_path):
        """
        Generate visualizations for integrated model results

        Args:
            revenue_at_risk_path (str): Path to revenue at risk data
            integrated_eval_path (str): Path to integrated evaluation results

        Returns:
            list: Paths to generated visualizations
        """
        try:
            logging.info("Generating integrated model visualizations")

            # Load data
            results_df = pd.read_csv(revenue_at_risk_path)

            with open(integrated_eval_path, "r") as f:
                integrated_metrics = json.load(f)

            generated_visuals = []

            # Plot 1: Scatter plot of churn probability vs revenue
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(
                results_df["churn_probability"],
                results_df["estimated_annual_revenue"],
                c=results_df["actual_churn"],
                s=50,
                alpha=0.6,
                cmap="coolwarm",
            )

            plt.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7)
            plt.colorbar(scatter, label="Actual Churn")
            plt.xlabel("Predicted Churn Probability", fontsize=12)
            plt.ylabel("Estimated Annual Revenue ($)", fontsize=12)
            plt.title("Churn Risk vs. Revenue", fontsize=14, fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            risk_revenue_path = os.path.join(
                self.visualizer_config.integrated_viz_dir, "risk_vs_revenue.png"
            )
            plt.savefig(risk_revenue_path)
            plt.close()
            generated_visuals.append(risk_revenue_path)

            # Plot 2: Revenue at risk by segment heatmap
            plt.figure(figsize=(10, 8))

            # Create risk matrix from the dataframe
            risk_matrix = pd.crosstab(
                results_df["value_segment"],
                results_df["risk_category"],
                values=results_df["revenue_at_risk"],
                aggfunc="sum",
                normalize=False,
            )

            sns.heatmap(risk_matrix, annot=True, fmt=",.0f", cmap="YlOrRd", cbar=True)
            plt.title(
                "Revenue at Risk Matrix: Value Segment vs. Churn Risk",
                fontsize=14,
                fontweight="bold",
            )
            plt.xlabel("Churn Risk Category", fontsize=12)
            plt.ylabel("Customer Value Segment", fontsize=12)
            plt.tight_layout()

            risk_matrix_path = os.path.join(
                self.visualizer_config.integrated_viz_dir, "risk_matrix_revenue.png"
            )
            plt.savefig(risk_matrix_path)
            plt.close()
            generated_visuals.append(risk_matrix_path)

            # Plot 3: Customer count by segment heatmap
            plt.figure(figsize=(10, 8))

            # Create risk matrix by count
            risk_matrix_count = pd.crosstab(
                results_df["value_segment"],
                results_df["risk_category"],
                normalize=False,
            )

            sns.heatmap(risk_matrix_count, annot=True, fmt="d", cmap="Blues", cbar=True)
            plt.title(
                "Customer Count Matrix: Value Segment vs. Churn Risk",
                fontsize=14,
                fontweight="bold",
            )
            plt.xlabel("Churn Risk Category", fontsize=12)
            plt.ylabel("Customer Value Segment", fontsize=12)
            plt.tight_layout()

            count_matrix_path = os.path.join(
                self.visualizer_config.integrated_viz_dir, "risk_matrix_count.png"
            )
            plt.savefig(count_matrix_path)
            plt.close()
            generated_visuals.append(count_matrix_path)

            # Plot 4: Distribution of revenue at risk
            plt.figure(figsize=(10, 6))
            sns.histplot(
                results_df["revenue_at_risk"], bins=30, kde=True, color="#e85d04"
            )
            plt.axvline(
                x=results_df["revenue_at_risk"].mean(),
                color="navy",
                linestyle="--",
                label=f"Mean: ${results_df['revenue_at_risk'].mean():.2f}",
            )
            plt.xlabel("Revenue at Risk ($)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.title("Distribution of Revenue at Risk", fontsize=14, fontweight="bold")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            revenue_dist_path = os.path.join(
                self.visualizer_config.integrated_viz_dir,
                "revenue_at_risk_distribution.png",
            )
            plt.savefig(revenue_dist_path)
            plt.close()
            generated_visuals.append(revenue_dist_path)

            logging.info(
                f"Generated {len(generated_visuals)} integrated model visualizations"
            )
            return generated_visuals

        except Exception as e:
            logging.error(f"Exception occurred during integrated visualization: {e}")
            raise CustomException(e, sys)
