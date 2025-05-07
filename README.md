# Credit Card Customer Analytics Framework

A comprehensive modular framework for predicting customer churn, assessing revenue impact, and identifying customer segments for targeted retention strategies.

## Project Overview

This framework provides a complete end-to-end solution for credit card customer analytics through three complementary modeling components:

1. **Churn Prediction (Classification)**: Predicts which customers are likely to discontinue their credit card service
2. **Revenue Impact Assessment (Regression)**: Quantifies the potential revenue loss from customer attrition
3. **Customer Segmentation (Clustering)**: Identifies distinct customer segments to enable targeted retention strategies

## Project Structure

```
credit-card-analytics/
├── config/                    # Configuration files
│   └── config.yaml            # Main configuration
├── artifacts/                 # Model outputs and artifacts (created at runtime)
├── logs/                      # Log files (created at runtime)
├── src/                       # Source code
│   ├── components/
│   │   ├── data_ingestion.py  # Data loading and splitting
│   │   ├── data_transformation.py  # Feature engineering and preprocessing
│   │   ├── model_trainer.py   # Model training and tuning
│   │   ├── model_evaluator.py # Model evaluation
│   │   └── visualizer.py      # Visualization generation
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py  # End-to-end training orchestration
│   │   └── predict_pipeline.py # Prediction pipeline
│   │
│   ├── exception.py           # Custom exception handling
│   ├── logger.py              # Logging functionality
│   └── utils.py               # Utility functions
│
└── main.py                    # Main executable
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-card-analytics.git
cd credit-card-analytics
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the models with default settings:

```python
from src.pipeline.train_pipeline import TrainingPipeline

# Initialize and run the pipeline
pipeline = TrainingPipeline()
results = pipeline.run(data_path="path/to/your/data.csv")
```

Or run from the command line:

```bash
python main.py --mode train --data_path path/to/your/data.csv
```

### Making Predictions

```python
from src.pipeline.predict_pipeline import PredictionPipeline
import pandas as pd

# Load data to predict
data = pd.read_csv("path/to/new_customers.csv")

# Initialize prediction pipeline
pipeline = PredictionPipeline()

# Generate predictions
predictions = pipeline.predict(data)

# Save predictions
pipeline.save_predictions(predictions, "path/to/save/predictions.csv")
```

Or run from the command line:

```bash
python main.py --mode predict --data_path path/to/new_customers.csv --output_path path/to/predictions.csv
```

## Input Data Format

The expected data format includes the following key features:

- `customer_age`: Age of the customer
- `gender`: Customer gender
- `dependent_count`: Number of dependents
- `education_level`: Education level
- `marital_status`: Marital status
- `income_category`: Income bracket
- `card_category`: Card type (Blue, Silver, Gold, etc.)
- `months_on_book`: Months as a customer
- `total_relationship_count`: Number of products held
- `months_inactive_12_mon`: Months of inactivity
- `contacts_count_12_mon`: Contacts with bank
- `credit_limit`: Credit limit
- `total_revolving_bal`: Revolving balance
- `avg_utilization_ratio`: Average utilization ratio
- `total_trans_amt`: Total transaction amount
- `total_trans_ct`: Total transaction count
- `total_ct_chng_q4_q1`: Change in transaction count (Q4 over Q1)
- `total_amt_chng_q4_q1`: Change in transaction amount (Q4 over Q1)

For training data, additional features required:
- `churn_flag`: Whether the customer churned (1) or not (0)
- `est_annual_revenue`: Estimated annual revenue

## Configuration

The `config/config.yaml` file allows customization of:

- Data paths
- Model parameters
- Preprocessing steps
- Output directories

## Outputs

The framework generates the following outputs in the `artifacts` directory:

- **Trained Models**: Serialized models for churn prediction, revenue prediction, and clustering
- **Preprocessors**: Serialized preprocessing pipelines
- **Evaluations**: Model performance metrics and statistics
- **Visualizations**: Performance charts, feature importance plots, and segment visualizations
- **Reports**: Detailed model reports and business insights

## Performance Metrics

### Churn Model
- ROC-AUC
- Accuracy, Precision, Recall
- F1 Score
- Confusion Matrix

### Revenue Model
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- Mean Absolute Percentage Error

### Customer Segmentation
- Silhouette Score
- Inertia
- Cluster Distribution

### Integrated Metrics
- Total Revenue at Risk
- Revenue at Risk by Segment
- Customer Distribution

## License

[MIT License](LICENSE)