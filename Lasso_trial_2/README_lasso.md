# Lasso Regression for Transport Cost Prediction 🚛📈

This repository is part of a larger machine learning project aimed at **predicting transport cost** efficiently.  
Here, the focus is on **Lasso Regression**, used as a baseline and interpretable model to identify the most impactful features.

---

## 📂 Project Overview

Lasso regression was implemented to build a **sparse linear model** that performs both **feature selection** and **regularization**.  
The main objective was to reduce overfitting, improve generalization, and understand which engineered features contribute most to cost prediction.

---

## ⚙️ Folder Structure

```
lasso/
├── betteralpha_hyper_lasso.py           # Fine-tuned alpha value for optimal regularization
├── gridsearchCV.py                      # Hyperparameter tuning using GridSearchCV
├── gridsearchCV_fixed.py                # Fixed alpha grid tuning
├── hyperparameter_tuning.py             # Experimental tuning pipeline
├── improved_lasso.py                    # Enhanced baseline model with better preprocessing
├── lasso_outliersremove.py              # Lasso with outlier removal for robust training
├── lasso_quantiletransform.py           # Data normalization with QuantileTransformer
├── lasso_robust_kfold.py                # K-Fold cross-validation for stable performance
├── onehotencoding_lasso.py              # Lasso model using one-hot encoded categorical features
├── onehotencoding_lasso_fixed.py        # Refined encoding setup with tuned regularization
├── lgb_catboost_elasticnet.py           # Comparative multi-model evaluation
├── lgb_catboost_lasso.py                # Combined Lasso + LightGBM/ CatBoost testing
├── testing_models_gridsearchCV.py       # Script for validating tuned models
```

---

## 🧠 Approach

### **1. Baseline**
- Implemented a simple **Lasso regression** using default parameters.  
- Served as an interpretable linear baseline before adding complexity.

### **2. Feature Engineering**
- Applied **OneHotEncoding** and **QuantileTransform** to scale and handle categorical features.  
- Removed outliers and normalized distributions for better model stability.

### **3. Hyperparameter Tuning**
- Used **GridSearchCV** to find the best value of `alpha`.  
- Optimized both overfitting and underfitting through multiple runs.

### **4. Robust Validation**
- Implemented **K-Fold Cross Validation** to ensure model reliability.  
- Compared performance across folds for consistency.

### **5. Comparative Analysis**
- Benchmarked against **LightGBM**, **CatBoost**, and **ElasticNet** models.  
- The hybrid approach demonstrated clear improvements in prediction accuracy and generalization.

---

## 📊 Key Improvements

| Stage | Model Variant | Improvement |
|:------|:---------------|:-------------|
| 1️⃣ | Baseline Lasso | Initial reference score |
| 2️⃣ | Lasso + OneHotEncoding | Better handling of categorical features |
| 3️⃣ | Lasso + Outlier Removal | More robust predictions |
| 4️⃣ | Lasso + Quantile Transform | Improved scaling and convergence |
| 5️⃣ | Lasso + Robust K-Fold | Reduced variance and higher reliability |

---

## 📈 Results Summary

- **Reduced Overfitting:** Lasso’s regularization removed noisy features.
- **Improved Interpretability:** Clear insight into top cost-driving factors.
- **Stable CV Performance:** Consistent RMSE across folds.
- **Comparative Edge:** Served as a solid baseline for tree-based model improvements.

---

## 🚀 Future Enhancements

- Integrate **PolynomialFeatures** for limited non-linear effects.  
- Experiment with **ElasticNet** to balance L1 and L2 regularization.  
- Create an **automated feature selection dashboard** for model explainability.

---
