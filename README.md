# **Telecom Customer Churn Prediction Project** 📞📊

## **Overview**

This project focuses on predicting customer churn in the telecom industry using a dataset that includes demographic, account, and service information. The primary goal is to identify at-risk customers to implement proactive retention strategies, reducing churn-related revenue losses.

## **Table of Contents**

1. [Problem Statement](#problem-statement)  
2. [Dataset Overview](#dataset-overview)  
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
4. [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)  
5. [Model Development and Evaluation](#model-development-and-evaluation)  
6. [Prediction System](#prediction-system)  
7. [Insights and Recommendations](#insights-and-recommendations)  
8. [Conclusion and Next Steps](#conclusion-and-next-steps)

## **Problem Statement**

**Goal:**  
Predict customer churn based on historical data to help retain at-risk customers.  

**Business Impact:**  
- Reduce customer churn to increase customer lifetime value (CLV).  
- Implement targeted retention strategies for high-risk customers.  

## **Dataset Overview**

The dataset contains features related to customer demographics, account information, and service usage.  
- **Target Variable:** `Churn` (binary: Yes/No)  
- **Key Features:**  
  - **Demographic Data:** Gender, SeniorCitizen, Partner, Dependents  
  - **Account Information:** Contract, PaymentMethod, MonthlyCharges, TotalCharges  
  - **Service Data:** InternetService, OnlineBackup, DeviceProtection  

**Steps in Data Cleaning:**  
1. Converted `TotalCharges` to numeric and imputed missing values.  
2. Handled duplicate records.  
3. Capped outliers using the interquartile range (IQR) method.

## **Exploratory Data Analysis (EDA)**

### **Key Findings:**
1. **Churn Distribution:**  
   - Approximately 27% of customers churned, indicating class imbalance.  

2. **Influential Features:**  
   - **MonthlyCharges:** Higher monthly charges correlated with increased churn rates.  
   - **Tenure:** Customers with shorter tenure were more likely to churn.  
   - **Contract Type:** Month-to-month contracts had the highest churn rates.

**Visualizations:**  
- Count plot of churn distribution.  
- Box plot of `MonthlyCharges` vs. `Churn`.  
- Heatmap showing correlations between numerical features.

## **Data Preprocessing and Feature Engineering**

1. **Encoding Categorical Variables:**  
   - Used `LabelEncoder` for categorical features. Saved encoders for future use.  

2. **Feature Engineering:**  
   - Created a new feature `TenureGroup` to categorize customers based on tenure.  

3. **Scaling Numerical Features:**  
   - Applied `MinMaxScaler` to normalize `tenure`, `MonthlyCharges`, and `TotalCharges`.  

4. **Addressing Class Imbalance:**  
   - Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset.

## **Model Development and Evaluation**

### **Models Evaluated:**
1. **Logistic Regression**  
2. **Random Forest Classifier**  

### **Performance Metrics:**
| Model               | Accuracy  | ROC-AUC   |---|
| Logistic Regression | 74.94%    | 0.8344    |
| Random Forest       | 99.74%    | 0.9999    |

- **Random Forest** was selected as the best-performing model due to its exceptional accuracy and ROC-AUC score.  

## **Prediction System**

A **prediction system** was implemented to provide real-time churn predictions:  
- **Inputs:** Customer data including tenure, monthly charges, contract type, and more.  
- **Outputs:**  
  - Prediction: `Churn` or `No Churn`.  
  - Probability: Likelihood of churn (e.g., 66% for `Churn`).  

### **Sample Predictions:**
| Customer  | Prediction  | Probability |--|
| Customer 1| No Churn    | 48.41%      |
| Customer 2| No Churn    | 3.20%       |
| Customer 3| No Churn    | 49.12%      |
| Customer 4| Churn       | 66.29%      |

## **Insights and Recommendations**

### **Insights from Model Evaluation:**
1. **High Performance:**  
   - Random Forest demonstrated near-perfect accuracy (**99.74%**) and ROC-AUC (**0.9999**).  
2. **Churn Characteristics:**  
   - High churn probability is associated with customers who:  
     - Have short tenure.  
     - Use month-to-month contracts.  
     - Pay higher monthly charges.  

### **Recommendations:**
1. **Retention Strategies:**  
   - Offer discounts or loyalty programs for month-to-month customers.  
   - Implement personalized retention campaigns for customers with high churn probabilities.  

2. **Model Deployment:**  
   - Deploy the **Random Forest model** for real-time churn prediction.  
   - Use probability thresholds to prioritize high-risk customers for retention efforts.

## **Conclusion and Next Steps**

### **Conclusion:**
- The project successfully identified high-risk customers using Random Forest, achieving exceptional predictive performance.  
- Insights from EDA revealed actionable patterns in customer churn behavior.  

### **Next Steps:**
1. **Deploy the model in production** to enable real-time churn predictions.  
2. **Monitor model performance** and retrain with updated data to maintain accuracy.  
3. Develop a **dashboard** to visualize churn trends and prediction outcomes.  

## **How to Run the Project**

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/telco-churn-prediction.git
   cd telco-churn-prediction
   ```

2. Install required libraries:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the prediction system:  
   ```python
   python predict_churn.py
   ```

Feel free to explore the project and share your feedback! 🚀  
📧 **Contact:** maqbuul@outlook.com 

#CustomerChurn #DataScience #MachineLearning #TelecomIndustry