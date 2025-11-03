# CatBoost Model for Transport Cost Prediction 🚚🔥

This repository is part of the broader machine learning pipeline for **predicting transport costs**.  
The focus here is on leveraging **CatBoost**, a high-performance gradient boosting algorithm known for handling categorical data efficiently and reducing the need for manual preprocessing.

---

## 📂 Project Overview

CatBoost was used to capture **non-linear feature interactions** and **boost model interpretability** without heavy feature engineering.  
Compared to linear models like Lasso, CatBoost provides superior accuracy, especially when categorical and numerical data interact in complex ways.

---

## ⚙️ Folder Structure

```
catboost/
├── catboost/
│   ├── foldsincreased_catboost.py          # CatBoost model with more cross-validation folds
│   ├── groupkfold_catboost.py              # Group K-Fold validation for structured data
│   ├── groupkfold_catboost_modified.py     # Modified validation for improved stability
│   ├── lgb_with_catboost.py                # Combined LightGBM + CatBoost ensemble
│   ├── lgb_with_catboost_modified.py       # Tuned hybrid model for better synergy
│   ├── lgb_with_catboost_modified2.py      # Final optimized hybrid model
```

---

## 🧠 Approach

### **1. Baseline CatBoost**
- Trained using default parameters and dataset splits.  
- Served as the non-linear baseline for transport cost prediction.

### **2. Handling Categorical Features**
- CatBoost’s native categorical handling reduced preprocessing overhead.  
- Automatically learned embeddings for categorical data.

### **3. Validation Strategy**
- Implemented **Group K-Fold** to preserve dependencies among related samples.  
- Created **fold-increased experiments** to test model robustness.

### **4. Hybrid Models**
- Combined **CatBoost + LightGBM** models to create a more balanced ensemble.  
- The hybrid achieved stronger generalization than either model alone.

### **5. Optimization**
- Tuned key parameters: `learning_rate`, `depth`, and `iterations`.  
- Evaluated using MAE and RMSE for consistent scoring across all folds.

---

## 📊 Key Improvements

| Stage | Model Variant | Improvement |
|:------|:---------------|:-------------|
| 1️⃣ | Baseline CatBoost | Strong nonlinear baseline |
| 2️⃣ | GroupKFold CatBoost | Better validation consistency |
| 3️⃣ | Fold-Increased CatBoost | Improved robustness and stability |
| 4️⃣ | CatBoost + LightGBM | Synergistic performance boost |
| 5️⃣ | Tuned Hybrid Model | Best generalization and accuracy |

---

## 📈 Results Summary

- **Higher accuracy** compared to Lasso and baseline LightGBM models.  
- **Improved robustness** with Group K-Fold validation.  
- **Seamless categorical handling** — no need for one-hot encoding.  
- **Hybrid ensemble** achieved the lowest MAE and RMSE among all models.

---

## 🚀 Future Enhancements

- Integrate **CatBoost’s feature importance visualization** to analyze key drivers.  
- Explore **multi-objective optimization** for time–cost tradeoffs.  
- Deploy hybrid CatBoost–LGBM model as a **production-ready API**.

---
