# Medical Equipment Transport Cost Prediction — Boosting Model

This project implements a **machine learning regression pipeline** to predict **medical equipment transport costs** using **Gradient Boosting Regression**. The notebook covers **data preprocessing, feature engineering, model training, hyperparameter tuning, validation, and submission generation** — all automated via a Scikit-learn pipeline.

---

## Overview

The goal of this project is to develop a predictive model that estimates the **TransportCost** based on various equipment, supplier, and delivery attributes. The pipeline ensures reproducible preprocessing and streamlined model deployment.

---

## Table of Contents
1. [Setup and Imports](#setup-and-imports)  
2. [Data Loading](#data-loading)  
3. [Preprocessing](#preprocessing)  
4. [Modeling Approach](#modeling-approach)  
5. [Training and Validation](#training-and-validation)  
6. [Evaluation Metrics](#evaluation-metrics)  
7. [Test Predictions and Submission](#test-predictions-and-submission)  
8. [Dependencies](#dependencies)  

---

## Setup and Imports

All standard Python libraries for machine learning are used:

```
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
```

---

## Data Loading

- Training and testing datasets are loaded using `pandas.read_csv()`.
- Date columns (`OrderPlacedDate`, `DeliveryDate`) are parsed as datetime.
- Dataset shapes:  
  - **Train:** (5000, 20 columns)  
  - **Test:** (500, 19 columns)

Both contain attributes such as equipment dimensions, supplier reliability, hospital location, and delivery settings.

---

## Preprocessing

Data preprocessing is performed within an automated **ColumnTransformer** pipeline:

### Key Steps
- **Date Handling:**  
  Dates are converted to datetime, and a new feature `deliverydelay` is created as the number of days between `DeliveryDate` and `OrderPlacedDate`.

- **Yes/No Encoding:**  
  Boolean columns like `CrossBorderShipping`, `UrgentShipping`, `InstallationService`, `FragileEquipment`, and `RuralHospital` are converted to binary (1 for Yes, 0 for No).

- **Feature Cleaning:**  
  - Dropped text-based and identifier columns (`HospitalLocation`, `HospitalInfo`) that are not numeric or categorical features.
  - `HospitalId` is retained only in the test set for submission ID mapping.

- **Feature Separation:**  
  - Numeric columns: continuous and binary numeric features (e.g., `SupplierReliability`, `EquipmentHeight`, `BaseTransportFee`, `deliverydelay`).
  - Categorical columns: nominal or text features (e.g., `HospitalId`, `SupplierName`, `EquipmentType`, `TransportMethod`).

### Pipelines Used
- Numeric pipeline: `SimpleImputer(strategy='median')`  
- Categorical pipeline:  
  `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`

These are combined in a `ColumnTransformer`, enabling consistent transformations for both training and inference.

---

## Modeling Approach

The final model uses **Gradient Boosting Regression** via Scikit-learn’s implementation.

### Model Configuration
After grid search tuning:
- `n_estimators = 200`
- `learning_rate = 0.1`
- `max_depth = 3`
- `min_samples_split = 2`
- `random_state = 42`


### Hyperparameter Tuning
A grid search (`GridSearchCV`) was conducted on:
- `n_estimators`: [50, 100, 200]  
- `learning_rate`: [0.01, 0.1, 0.2]  
- `max_depth`: [3, 5, 7]  
- `min_samples_split`: [2, 5, 10]

Best parameters achieved from 3-fold cross-validation:


---

## Evaluation Metrics

Performance on the validation set:

- **Mean Squared Error (MSE):** 1,263,502,879.8257  
- **R² Score:** 0.4304  

These metrics indicate reasonable predictive performance given potential feature variability and noise.

---

## Test Predictions and Submission

- Predictions are generated using the fitted pipeline on `X_test`.
- Results are saved as:


Output file: **`boosting_submission.csv`**

---

## Dependencies

Ensure the following are installed:

| Library | Version |
|----------|----------|
| Python | 3.12.5 |
| pandas | ≥1.5.0 |
| scikit-learn | ≥1.6 |
| numpy | ≥1.24 |


---

## Summary

This notebook demonstrates a fully automated machine learning pipeline for tabular data regression:
- Feature engineering (including date-based delay feature)
- Robust imputation and encoding
- Gradient boosting tuning via GridSearchCV
- Integrated preprocessing and model workflow in a single pipeline
- Ready-to-submit predictions


