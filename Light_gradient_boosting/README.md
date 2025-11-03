 
---

## 🧠 Problem Statement

Predict the transportation cost based on given input parameters such as shipment details, distance, weight, or other operational features.  
The objective is to build a model that accurately predicts cost while avoiding overfitting and ensuring consistent results across datasets.

---

## ⚙️ Approach

### 1. Baseline Model
- Used LightGBM Regressor with default parameters.
- Performed minimal preprocessing and basic feature encoding.
- Served as the initial benchmark for comparison.

### 2. Hybrid Model
- Integrated ElasticNet regularization to stabilize model weights.
- Helped reduce variance and improved interpretability.

### 3. Improved Model with K-Fold Cross-Validation
- Implemented KFold to ensure the model generalizes well across data splits.
- Evaluated model stability and reduced dependence on random train-test splits.
- Tuned hyperparameters such as
  - `num_leaves`
  - `learning_rate`
  - `min_data_in_leaf`
  - `feature_fraction`
- Used Mean Absolute Error (MAE) as the primary evaluation metric.

---

## 📈 Model Improvement Summary

 Version  Technique Used  Key Feature  MAE (↓)  Improvement 
----------------------------------------------------------------
 Baseline  LightGBM  Default params  5.28e+09  — 
 v2  LightGBM + ElasticNet  Regularization  4.82e+09  +8.7% 
 v3  LightGBM + KFold  Generalization  4.46e+09  +15.5% 

 Result The enhanced LightGBM with K-Fold validation showed consistent and significant improvement over baseline performance.

---

## 📊 Insights

- LightGBM outperformed linear models due to its ability to handle non-linear relationships in features.  
- K-Fold validation confirmed model reliability and reduced variance across folds.  
- Regularization (ElasticNet) improved convergence and mitigated overfitting.

---

## 🧩 Key Learnings

- Importance of cross-validation in performance assessment.
- Trade-off between model complexity and generalization.
- Role of feature selection and encoding in regression accuracy.
- Combining linear and tree-based models can yield hybrid strengths.

---

## 🚀 Future Improvements

- Feature importance analysis and selection.
- Hyperparameter optimization using Optuna or Bayesian search.
- Adding ensemble stacking between LightGBM and ElasticNet.
- Deployment-ready version (Flask  FastAPI).


## 🏷️ Tags
`Machine Learning` `LightGBM` `ElasticNet` `Transport Cost` `Regression` `KFold` `Optimization`
