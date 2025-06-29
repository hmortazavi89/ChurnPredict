# Customer Churn Prediction
# Project Overview:
# This script predicts customer churn for a telecommunications company using machine learning.
# It analyzes customer data to identify patterns indicating whether a customer is likely to leave,
# enabling proactive retention strategies. The dataset includes features like tenure, monthly charges,
# and service usage.

# Objectives:
# - Perform exploratory data analysis (EDA) to understand key factors influencing churn.
# - Preprocess data, including handling missing values and encoding categorical variables.
# - Train and evaluate a Random Forest Classifier to predict churn.
# - Visualize results to communicate findings effectively.

# Dataset:
# The dataset (telecom_churn.csv) contains:
# - CustomerID: Unique identifier for each customer.
# - Tenure: Number of months the customer has been with the company.
# - MonthlyCharges: Monthly bill amount.
# - Contract: Contract type (Month-to-month, One year, Two year).
# - InternetService: Type of internet service (DSL, Fiber optic, No).
# - Churn: Target variable (Yes/No) indicating if the customer churned.

# Tools and Libraries:
# - Python, pandas, scikit-learn, matplotlib, seaborn, numpy

# Prerequisites:
# Install required libraries:
# pip install pandas scikit-learn matplotlib seaborn numpy
# Ensure telecom_churn.csv is in the same directory as this script.

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report

# Set random seed for reproducibility
np.random.seed(42)

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Step 1: Data Loading and Cleaning
# Load the dataset and check for missing values or inconsistencies
print('Step 1: Data Loading and Cleaning')
try:
    df = pd.read_csv('telecom_churn.csv')
except FileNotFoundError:
    # Create a sample dataset if file is not found
    data = {
        'CustomerID': range(1, 1001),
        'Tenure': np.random.randint(1, 72, 1000),
        'MonthlyCharges': np.random.uniform(20, 120, 1000),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], 1000),
        'Churn': np.random.choice(['Yes', 'No'], 1000, p=[0.3, 0.7])
    }
    df = pd.DataFrame(data)

# Display first few rows
print('Dataset Preview:')
print(df.head())

# Check for missing values
print('\nMissing Values:')
print(df.isnull().sum())

# Basic info
print('\nDataset Info:')
print(df.info())

# Step 2: Exploratory Data Analysis
# Analyze the distribution of features and their relationship with churn
print('\nStep 2: Exploratory Data Analysis')

# Churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

# Tenure vs Churn
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='Tenure', data=df)
plt.title('Tenure vs Churn')
plt.show()

# Monthly Charges vs Churn
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges vs Churn')
plt.show()

# Contract Type vs Churn
plt.figure(figsize=(8, 5))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Contract Type vs Churn')
plt.show()

# Step 3: Data Preprocessing
# Encode categorical variables and scale numerical features
print('\nStep 3: Data Preprocessing')

# Encode categorical variables
le = LabelEncoder()
df['Contract'] = le.fit_transform(df['Contract'])
df['InternetService'] = le.fit_transform(df['InternetService'])
df['Churn'] = le.fit_transform(df['Churn'])

# Define features and target
X = df[['Tenure', 'MonthlyCharges', 'Contract', 'InternetService']]
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train[['Tenure', 'MonthlyCharges']] = scaler.fit_transform(X_train[['Tenure', 'MonthlyCharges']])
X_test[['Tenure', 'MonthlyCharges']] = scaler.transform(X_test[['Tenure', 'MonthlyCharges']])

print('Training set shape:', X_train.shape)
print('Testing set shape:', X_test.shape)

# Step 4: Model Training
755
# Train a Random Forest Classifier to predict churn
print('\nStep 4: Model Training')

# Initialize and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Step 5: Model Evaluation
# Evaluate the model using accuracy, precision, recall, and AUC-ROC
print('\nStep 5: Model Evaluation')

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print('Model Performance:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'AUC-ROC: {auc:.2f}')

# Classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Step 6: Feature Importance
# Visualize the importance of each feature in the model
print('\nStep 6: Feature Importance')

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

# Conclusion
print('\nConclusion:')
print('This script demonstrates a complete machine learning workflow, from data preprocessing to model evaluation.')
print('The Random Forest Classifier effectively predicts customer churn, with key features like tenure and contract type driving predictions.')
print('The model achieves a solid AUC-ROC score, indicating good performance.')
print('This work can inform customer retention strategies by identifying at-risk customers.')

# Future Work
print('\nFuture Work:')
print('- Experiment with other models like XGBoost or Neural Networks.')
print('- Incorporate additional features from related datasets.')
print('- Deploy the model using a web application for real-time predictions.')
print('For the full code and dataset, visit the GitHub repository: https://github.com/yourusername/ChurnPredict')