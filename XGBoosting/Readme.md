# XGBoost Model for Medical Equipment Cost Prediction

This repository implements a **machine learning regression pipeline** based on **XGBoost Regressor** to predict the **TransportCost** of medical equipment shipments. It includes all steps from **data loading**, **preprocessing**, **feature engineering**, **model training**, **hyperparameter tuning**, and **final submission** generation.

---

## Overview

The project’s objective is to develop an accurate regression model for predicting transport costs based on various hospital, supplier, and shipment attributes. The model leverages the efficiency of **XGBoost**, an optimized gradient boosting framework.

---

## Table of Contents

1. [Setup and Imports](#setup-and-imports)  
2. [Data Loading](#data-loading)  
3. [Feature Engineering](#feature-engineering)  
4. [Preprocessing Pipeline](#preprocessing-pipeline)  
5. [Model and Hyperparameter Tuning](#model-and-hyperparameter-tuning)  
6. [Training and Validation](#training-and-validation)  
7. [Evaluation Metrics](#evaluation-metrics)  
8. [Test Prediction and Submission](#test-prediction-and-submission)  
9. [Dependencies](#dependencies)  

---

## Setup and Imports

Core libraries used:

```
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import xgboost as xgb

```


These libraries enable data manipulation, model training, and evaluation with integrated preprocessing through pipelines.

---

## Data Loading

- **Training and test datasets** are loaded using `pandas.read_csv()`.
- Date columns (`OrderPlacedDate`, `DeliveryDate`) are parsed as datetime for feature extraction.
- Dataset shapes:  
  - **Train:** (5000, 20 columns)  
  - **Test:**  (500, 19 columns)
  
Example features:
`SupplierReliability`, `EquipmentHeight`, `EquipmentWeight`, `EquipmentType`, `TransportMethod`, `UrgentShipping`, `DeliveryDate`, `HospitalId`, etc.

---

## Feature Engineering

Several derived features were created and invalid entries handled properly:

### 1. Date Correction and Duration
A custom function ensures proper ordering of `OrderPlacedDate` and `DeliveryDate`:
- Swaps dates if `OrderPlacedDate` > `DeliveryDate`.
- Calculates `deliverydelay = (DeliveryDate - OrderPlacedDate)` in days.
- Fills missing or zero delays with the **mean positive delay value**.

### 2. Boolean to Binary Conversion
Yes/No categorical columns are converted to integers:
- Columns: `CrossBorderShipping`, `UrgentShipping`, `InstallationService`, `FragileEquipment`, `RuralHospital`.
- Mapping: Yes → 1, No → 0.

### 3. Temporal Features
Extracted from order and delivery dates:
- `ordermonth`, `orderweekday`, `deliverymonth`, and `deliveryweekday`.

### 4. Target Column Cleaning
- Negative and zero `TransportCost` values are replaced:
  - Negative → 0
  - Zero → mean of positive costs.

---

## Preprocessing Pipeline

Feature columns are classified as **numeric** and **categorical**:

```
numeric_features = ['SupplierReliability', 'EquipmentHeight', 'EquipmentWidth',
'EquipmentWeight', 'EquipmentValue', 'BaseTransportFee',
'CrossBorderShipping', 'UrgentShipping', 'InstallationService',
'FragileEquipment', 'RuralHospital', 'deliverydelay',
'ordermonth', 'orderweekday', 'deliverymonth', 'deliveryweekday']

categorical_features = ['HospitalId', 'SupplierName', 'EquipmentType', 'TransportMethod']

```


### Transformation Pipelines

- **Numeric:** Missing values replaced using `SimpleImputer(strategy='median')`
- **Categorical:** Imputation with mode followed by one-hot encoding  
  (`OneHotEncoder(handle_unknown='ignore')`)

Combined into a `ColumnTransformer` for seamless integration with the model.

---

### Hyperparameter Search

Performed using `GridSearchCV` (3-fold cross-validation):

| Parameter | Range Tested |
|------------|---------------|
| n_estimators | [100, 200, 300] |
| learning_rate | [0.05, 0.1, 0.2] |
| max_depth | [3, 5, 6, 7] |
| subsample | [0.7, 1.0] |

**Best parameters:**

```
n_estimators = 300
learning_rate = 0.2
max_depth = 3
subsample = 1.0
```


Validation metrics:
- **MSE:** 1,208,630,987.9850  
- **R²:** 0.4425

---

## Training and Validation

Data is split for validation:
```
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)     
```

Final metrics:
- **Validation MSE:** 1,208,630,987.9850  
- **R²:** 0.4425  

Model performed similarly to Gradient Boosting but with faster training speed and better handling of complex nonlinearities.

---

## Test Prediction and Submission

Predictions are made using the trained pipeline:
```
test_preds = model.predict(X_test)
submission = pd.DataFrame({
'HospitalId': test['HospitalId'],
'TransportCost': test_preds
})
submission.to_csv('xgboost_submission.csv', index=False)
```

Output file: **`xgboost_submission.csv`**

---

## Dependencies

| Library | Version |
|----------|----------|
| Python | 3.12.5 |
| pandas | ≥1.5.0 |
| scikit-learn | ≥1.6 |
| xgboost | ≥2.1 |
| numpy | ≥1.24 |


---

## Summary

This notebook automates the complete regression modeling process:
- Robust preprocessing and feature creation  
- Integrated pipeline with XGBoost regressor  
- Hyperparameter optimization through GridSearchCV  
- Performance evaluation with MSE and R²  
- Export-ready test predictions  

It provides a unified, reproducible framework for scalable tabular regression using boosting methods.

---


