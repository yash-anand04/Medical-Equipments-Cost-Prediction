# LASSO Regression Model for Medical Equipment Transport Cost Prediction

## Overview
This project focuses on predicting the **transportation cost of medical equipment to hospitals** using machine learning.  
The chosen model is **LASSO Regression (Least Absolute Shrinkage and Selection Operator)** — a linear regression technique that performs both **regularization** and **feature selection** by shrinking less influential feature coefficients to zero.  

Over multiple stages of refinement, the model evolved from a baseline LASSO approach on raw data to a fully optimized pipeline that includes **data preprocessing, feature engineering, and hyperparameter tuning** for improved performance and reliability.

---

## Model Evolution and Performance

### 1. Simple LASSO (No Preprocessing)
At first, the LASSO model was directly trained on the raw dataset without any preprocessing or feature cleaning.  
While it ran successfully, model performance was poor because of multiple data inconsistencies:
- Presence of **negative target values** (`Transport_Cost`).
- **Unprocessed date columns** in string format.
- **Unscaled numerical features** that distorted model coefficients.
- **Categorical values** that LASSO couldn’t interpret without encoding.

**Reason for poor performance:**  
LASSO assumes clean numeric input and regular feature scales. With noise, unencoded text, and extreme values, its weight penalties became ineffective, producing a **high RMSE (Root Mean Squared Error)** and unstable predictions.

---

### 2. LASSO with Cost Preprocessing Only
The next iteration attempted to clean the target variable (`Transport_Cost`):
- All **negative costs** were replaced with **0**.
- Then, all **zero values** were replaced with the **mean positive cost**.

This transformation stabilized the target distribution and eliminated unrealistic cost values.  
However, since the feature space was unchanged (no handling for categories, missing values, or date fields), model performance only modestly improved.

**Summary:**  
LASSO benefited from a cleaner target, but feature-side noise and inconsistent data still limited predictive accuracy.

---

### 3. LASSO with Cost + Date Preprocessing
In this version, cost preprocessing was maintained and **date cleaning** was introduced:
- Converted `Order_Placed_Date` and `Delivery_Date` to proper datetime types.
- **Swapped** entries where order date > delivery date.
- Created a new numeric feature, **Delivery_Duration = (Delivery_Date - Order_Placed_Date)**.
- Replaced invalid or missing durations with the mean delivery duration.

This gave LASSO a **time-based numerical feature** that captured logistical patterns.  
Model performance improved further as date inconsistencies were resolved and more meaningful time information fed into the model.

Still, certain limitations like **feature scaling, missing imputations, and categorical variable handling** prevented it from reaching optimal RMSE scores.

---

### 4. Final Model: Full Preprocessing + Hyperparameter Optimization
The final version implemented a **fully automated pipeline** that included:

#### a. Full Preprocessing
- **Negative values fixed** in target variable.
- **Date engineering** with duration feature added and inconsistent pairs swapped.
- **Feature transformations** via a `ColumnTransformer`:
  - Numeric columns: mean imputation + `StandardScaler`.
  - Categorical columns: mode imputation + `OneHotEncoder(handle_unknown='ignore')`.

#### b. Hyperparameter Optimization
- Used **GridSearchCV** to tune the LASSO regularization parameter `alpha`.
- Searched across `[0.0001, 0.001, 0.01, 0.1, 1, 10]` using 5-fold cross-validation.
- Selected the **best alpha** that minimized validation error.

#### c. Performance Evaluation
- Achieved the **lowest validation RMSE** among all attempts.
- Model generalized well on unseen test data.
- Final negative predictions were replaced with the mean of positive test predictions to ensure realistic cost outputs.

**Why This Version Works Best:**
- **Complete preprocessing** ensures all numeric and categorical features are properly scaled, encoded, and cleaned.
- **Cross-validation** avoids overfitting to train/validation splits.
- **Regularization tuning** finds the alpha that balances bias and variance.
- **Automated pipeline** ensures consistent transformations between train and test data.

---

## Pipeline Workflow
1. Load train and test datasets.
2. Fix invalid cost and date values.
3. Engineer `Delivery_Duration`.
4. Apply preprocessing (imputation, scaling, encoding).
5. Perform hyperparameter tuning via GridSearchCV.
6. Validate and compute RMSE.
7. Generate test predictions and replace negatives with mean positive value.
8. Export final submission CSV.

---

## Key Files
| File | Description |
|------|--------------|
| `train.csv` | Training dataset containing features and target (`Transport_Cost`). |
| `test.csv` | Test dataset used for final predictions. |
| `submission_optimized.csv` | Output file containing predicted transport costs per hospital. |
| `model_lasso_pipeline.py` | Python script implementing complete preprocessing, training, tuning, and prediction pipeline. |

---

## Results Summary
| Model Version | Preprocessing Applied | Date Handling | Hyperparameter Tuning | RMSE (Relative) |
|----------------|----------------------|----------------|------------------------|-----------------|
| Simple LASSO | None | No | No | Very High |
| LASSO + Cleaned Cost | Target Cleaning | No | No | High |
| LASSO + Cleaned Cost + Dates | Target + Feature Cleaning | Yes | No | Medium |
| **LASSO + Full Preprocessing + Tuning (Final)** | Full Numerical + Categorical | Yes | Yes | **Lowest** |

---

## Running the Model
1. Ensure `train.csv` and `test.csv` are in the same directory as the Python script.
2. Run the pipeline:
