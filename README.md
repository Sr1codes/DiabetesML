# Predicting Diabetes Risk using Machine Learning

This project is my final submission for my Machine Learning course, focused on building a machine learning model to predict diabetes risk using health and demographic data. My goal was to Help identify individuals at risk based on common indicators like blood glucose levels, BMI, and more

## Overview

Using a dataset of 10,000 samples with 21 features, I cleaned and preprocessed the data, engineered a target label (diabetic or not), and trained two models:

- **Logistic Regression** with class balancing
- **Random Forest Classifier** with hyperparameter tuning and cross-validation

The model predicts whether a person is likely to be diabetic based on medical and lifestyle attributes.

## Libraries Used

- `pandas`, `numpy`, `matplotlib`
- `scikit-learn`: for ML modeling, evaluation, and preprocessing

## Dataset Info

- **Source**: `diabetes_dataset.csv`
- **Size**: 10,000 rows × 21 columns
- **Target Label**: `Diabetes` (1 = Diabetic, 0 = Not Diabetic)

The label was created based on:
- `HbA1c ≥ 6.5%`
- OR `Fasting Blood Glucose ≥ 126 mg/dL`

##  Preprocessing Steps

- Shuffled dataset
- Handled categorical features with `OneHotEncoding`
- Scaled features using `StandardScaler`
- Handled class imbalance with `class_weight='balanced'` in models

## Models & Performance

### 1. Logistic Regression
- Accuracy: **94.6%**
- Strength: Simple, interpretable
- Weakness: Slightly worse at handling class imbalance

### 2. Random Forest (tuned)
- Accuracy: **~99.96%**
- Strength: Handles nonlinearities & imbalance
- Features ranked by importance

## Evaluation Reportsd

-  **Accuracy**
-  **Confusion Matrix**
-  **ROC Curve**
-  **Classification Report**
-  **Feature Importance Visualization**

## Results

- **Most important features**:
  - `HbA1c` (by far!)
  - `Fasting_Blood_Glucose`
- Lifestyle and demographic features like activity level or ethnicity had minor contributions
- Class imbalance was significant — over 90% of cases were diabetic, requiring careful handling/ balancing of the data, so the model doesnt favor one side which creates bias. 
