# KNN Regression for Medical Equipment Transport Cost Prediction

## Overview
This project addresses the prediction of medical equipment transport costs to hospitals using a K-Nearest Neighbors (KNN) regression model.  
The solution demonstrates the importance of data cleaning and feature engineering, especially for temporal (date-related) and target variable preprocessing, and compares KNN's effectiveness to that of optimized linear models.

---

## 1. Why Simple KNN Regression Falls Short

A basic KNN regression approach applied directly to the unprocessed data results in unreliable predictions due to:
- **Negative target values** (`Transport_Cost`) distorting cost distribution.
- **Date columns** not interpreted as durations but as object string types.
- Absence of systematic handling for missing/categorical values, and for feature scale disparities.

**Result:**  
The simple KNN model struggles with noisy, unscaled, and misrepresented data, leading to high RMSE (error) and subpar predictive accuracy.

---

## 2. KNN Regression with Cost and Date Preprocessing

To address these issues, several data cleaning and feature engineering steps were introduced:
- Set all negative costs to 0, then replaced any 0s with the mean of valid positive costs.
- Parsed `Order_Placed_Date` and `Delivery_Date` to datetime; swapped values if inconsistent (delivery before order).
- Engineered a numeric `Delivery_Duration` feature, with negative or missing durations replaced by the mean.
- Used a `ColumnTransformer` to:
  - Impute and standardize numeric features.
  - Impute and one-hot encode categorical features.

Model pipeline:
- Preprocessing combines all transformations above.
- Final estimator is `KNeighborsRegressor` (default `n_neighbors=5`).

**Result:**  
This cleaned and transformed KNN pipeline led to **significantly improved predictions** over the naïve version. Validation RMSE dropped as features became more meaningful and consistent.

---

## 3. Why KNN, Even with Good Data, Is Less Effective than Tuned LASSO/Ridge

Despite these gains, KNN regression (with standard hyperparameters) tends not to match the predictive accuracy and robustness of **LASSO or Ridge regression** with:
- **Comprehensive preprocessing**
- **Feature selection/regularization** (LASSO)
- **Hyperparameter tuning** (e.g., GridSearchCV for `alpha` or K)

Reasons:
- KNN is sensitive to the local structure of the data and can be affected by noise or irrelevant/noisy dimensions after encoding.
- Linear models with regularization generalize better and routinely outperform KNN in higher-dimensional, structured tabular data—especially with engineered features and model tuning.

**Conclusion:**  
While KNN with careful preprocessing is notably better than “raw” KNN, for the given logistical regression problem, **LASSO or Ridge with hyperparameter optimization and robust pipelines consistently achieve lower RMSE and more stable results**.

---

## Summary Table

| Model                                 | Preprocessing        | Hyperparameter Tuning | Relative RMSE   |
|----------------------------------------|---------------------|----------------------|-----------------|
| Simple KNN                             | None                | No                   | High            |
| KNN + Cost & Date Preprocessing        | Target + Date Clean | No                   | Medium          |
| LASSO/Ridge + Full Pipeline (Best)     | Full                | Yes                  | **Lowest**      |

---

## Usage

1. Ensure `train.csv` and `test.csv` are available in the project directory.
2. Run the KNN regression script:
