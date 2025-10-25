# Ridge Regression for Medical Equipment Cost Prediction

## Overview
This project predicts the cost required to transport medical equipment to hospitals using regression models.  
The focus is on demonstrating the impact of **feature engineering and preprocessing** within a Ridge regression pipeline, and comparing it to simpler approaches and advanced LASSO solutions.

---

## 1. Why Simple Ridge Regression is Insufficient

The baseline model applied Ridge regression directly on the raw dataset, with no special handling for data quality issues. This approach failed to deliver competitive results because:

- The target column (`Transport_Cost`) contained **negative values** and outliers.
- **Date fields** (`Order_Placed_Date`, `Delivery_Date`) were treated as uninformative strings, not as useful features.
- Missing values or categorical encoding were not systematically handled.
- Numeric features were not scaled, so feature magnitudes varied widely.

**Result:**  
The model captured noisy or irrelevant relationships, producing **high RMSE and unreliable predictions**.

---

## 2. Ridge Regression with Cost and Date Preprocessing

To improve the model, both **target cleaning** and **date feature engineering** were applied:

- All **negative transport costs** were set to 0, then replaced with the mean of valid positive costs.
- Both `Order_Placed_Date` and `Delivery_Date` were converted to datetime objects.
- For entries where the order date was after the delivery date, the fields were swapped for logical consistency.
- A new numeric feature, `Delivery_Duration`, representing the days between order and delivery, was created.
- Any missing or negative durations were replaced with the mean.

The resulting pipeline included:
- **Imputation** and **standard scaling** for numeric features.
- **Imputation and one-hot encoding** for categorical features.
- Complete modular preprocessing using `ColumnTransformer`.

**Result:**  
The Ridge model with these improvements **performed significantly better than the naïve model**. Validation RMSE was reduced, and the predicted costs better reflected realistic logistical timelines.

---

## 3. Why LASSO with Full Preprocessing and Hyperparameter Tuning is Superior

Despite these gains, Ridge regression — even with improved features — still falls short compared to an optimized LASSO pipeline with:

- **Full target, date, numeric, and categorical preprocessing**
- **Automated hyperparameter tuning** via GridSearchCV for regularization strength
- **Robust cross-validation** to avoid overfitting

LASSO’s L1 regularization inherently encourages sparsity, automatically selecting the most important features and ignoring irrelevant ones. Hyperparameter optimization ensures the best trade-off between model complexity and generalization.

**Result:**  
- LASSO with all preprocessing and tuned hyperparameters **delivered the lowest RMSE**.
- Predictions were more robust and interpretable.
- The automated pipeline ensured reliability across multiple feature types and future data updates.

---

## Summary Table

| Model                            | Preprocessing           | Hyperparameter Tuning | Relative RMSE   |
|-----------------------------------|------------------------|----------------------|-----------------|
| Simple Ridge                     | None                   | No                   | Very High       |
| Ridge + Cost & Date Preprocessing | Target + Date Cleaning | No                   | Medium          |
| **LASSO + Full Pipeline (Best)**  | Full + Engineered      | Yes                  | **Lowest**      |

---

## Usage

1. Place `train.csv` and `test.csv` in the project directory.
2. Run the Ridge pipeline script:
