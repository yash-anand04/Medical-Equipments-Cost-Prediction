# Linear and Polynomial Regression for Medical Equipment Cost Prediction

## Overview
This project aims to predict the cost of transporting medical equipment to hospitals using **linear regression** and **polynomial regression** models.  
The workflow demonstrates the impact of **systematic data cleaning, feature engineering**, and **model selection** on predictive performance.

---

## 1. Why Simple Linear Regression Fails

A raw linear regression model on the unprocessed dataset does not perform well because:
- The `Transport_Cost` target contains negative and noisy values.
- Date fields (`Order_Placed_Date`, `Delivery_Date`) are not usable in raw string format.
- Categorical values are unencoded.
- Numeric features are not properly scaled, leading to ill-conditioned models and unstable coefficients.

**Result**:  
Predictions are of poor quality, with high RMSE and interpretability issues.

---

## 2. Linear and Polynomial Regression with Cost and Date Preprocessing

To overcome these issues, the pipeline implements the following:
- **Negative target values** are set to 0 and then all 0s are replaced by the mean valid cost.
- **Order and delivery dates** are converted to datetime types; any pairs with `Order_Placed_Date` after `Delivery_Date` are swapped.
- **Delivery duration** is computed as a quantitative feature; negative or missing durations are substituted with the mean.
- A **ColumnTransformer** is used:
  - For numerics: missing value imputation + standardization.
  - For categoricals: mode imputation + one-hot encoding.

Two models are trained and evaluated:
- **Simple linear regression** on preprocessed features.
- **Polynomial regression (degree 2)** to capture basic nonlinearities.

**Result**:  
These approaches achieve **lower RMSE** than unprocessed linear models, as the model now leverages meaningful, scaled, and consistent features.

---

## 3. Why Regularized/Optimized Models Are Better

Even with date and cost preprocessing, **ordinary linear and polynomial regression** can have limitations:
- Prone to **overfitting** (especially polynomial regression) without regularization.
- Sensitive to irrelevant/noisy features, which can bias coefficients.
- Cannot perform feature selection or robustly handle multicollinearity.

By contrast, advanced pipelines using **LASSO or Ridge with full preprocessing and hyperparameter optimization**:
- Provide built-in regularization via alpha/lambda parameters.
- Select only relevant features (LASSO).
- Avoid overfitting in higher dimensions—even compared to polynomials.
- Achieve the **lowest RMSE** in validation and more robust, generalizable predictions.

---

## Summary Table

| Model                                    | Preprocessing               | Degree | Regularization | Relative RMSE |
|-------------------------------------------|-----------------------------|--------|---------------|---------------|
| Simple Linear Regression                  | None                        | 1      | No            | High          |
| Linear Regression + Cost/Date Processing  | Target/Date Clean           | 1      | No            | Medium        |
| Polynomial Regression (deg=2) + Preproc   | Target/Date Clean           | 2      | No            | Medium        |
| LASSO/Ridge (Tuned, Best)                 | Full + Engineered           | 1      | Yes           | **Lowest**    |

---

## Usage

1. Place `train.csv` and `test.csv` in the project directory.
2. Run the pipeline with:
