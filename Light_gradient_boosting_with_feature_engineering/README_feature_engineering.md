# 🧩 Advanced Feature Engineering for Transport Cost Prediction

This project builds on previous work in **transport cost prediction** by introducing systematic **feature engineering**, **outlier handling**, and **hyperparameter optimization** techniques.  
The primary goal is to enhance model performance and robustness using **LightGBM** as the base algorithm, combined with advanced preprocessing and tuning strategies.

---

## 📁 Project Structure

```
feature_engineering/
├── lgb_kfold_featureeng.py          # LightGBM with engineered features and K-Fold validation
├── lgb_kfold_featureeng_modified.py # Enhanced version with optimized feature pipeline
├── lgb_kfold_optuna_log.py          # Optuna-based hyperparameter tuning with logging
├── outlier_removal.py               # Script to detect and remove data outliers
```

---

## 🧠 Problem Statement

Predict the **transportation cost** using improved input features derived through advanced preprocessing.  
The objective is to show how **feature engineering and optimization** can substantially improve prediction accuracy over baseline models.

---

## ⚙️ Approach

### 1. **Baseline Setup**
- Started with previously validated LightGBM pipeline (from earlier project).
- Used same dataset and preprocessing to ensure fair comparison.

### 2. **Feature Engineering**
- Created **derived features** (ratios, aggregates, categorical encodings).
- Applied **log transformations** to reduce skewness.
- Standardized numerical values for balanced scaling.
- Encoded categorical variables with high cardinality using frequency and target encoding.

### 3. **Outlier Handling**
- Implemented an **outlier_removal.py** module.
- Detected anomalies using **IQR (Interquartile Range)** and **Z-score** methods.
- Removed or clipped extreme values to stabilize model learning.

### 4. **Cross-Validation & Model Training**
- Used **K-Fold Cross-Validation** for reliable performance estimation.
- Trained **LightGBM Regressor** with fine-tuned parameters.
- Compared baseline, engineered, and optimized versions.

### 5. **Hyperparameter Optimization**
- Utilized **Optuna** (`lgb_kfold_optuna_log.py`) for Bayesian search.
- Tuned:
  - `num_leaves`
  - `max_depth`
  - `learning_rate`
  - `feature_fraction`
  - `lambda_l1`, `lambda_l2`
- Achieved best parameters automatically logged for reproducibility.

---

## 📈 Model Performance Summary

| Version | Technique | Key Enhancement | MAE (↓) | Improvement |
|----------|------------|----------------|----------|--------------|
| Baseline | LightGBM (previous project) | Default features | 4.46e+09 | — |
| v2 | + Feature Engineering | Derived + encoded features | 3.88e+09 | +13.0% |
| v3 | + Outlier Removal | Stabilized data | 3.71e+09 | +17.0% |
| v4 | + Optuna Optimization | Auto-tuned hyperparams | 3.55e+09 | +20.4% |

> **Result:** Feature engineering, outlier handling, and parameter tuning significantly improved prediction accuracy and model generalization.

---

## 📊 Insights

- **Feature quality** directly impacts LightGBM performance more than sheer model complexity.  
- **Outlier control** prevents bias and variance explosions in regression tasks.  
- **Optuna-based tuning** ensures efficient exploration of hyperparameter space.  
- Combining these improvements yields a strong, generalizable transport cost prediction model.

---

## 🧩 Key Learnings

- Feature engineering is as crucial as model selection.  
- Outlier management helps stabilize training and prevent overfitting.  
- Automated optimization frameworks like Optuna save time and increase reproducibility.  
- Cross-validation ensures consistent performance evaluation.

---

## 🚀 Future Work

- Integrate **SHAP** or **LIME** for model interpretability.  
- Automate feature selection using **Recursive Feature Elimination (RFE)**.  
- Explore ensemble stacking with **CatBoost** or **XGBoost**.  
- Build an end-to-end ML pipeline for deployment.

---

