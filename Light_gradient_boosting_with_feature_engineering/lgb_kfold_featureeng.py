# transport_cost_tuned.py

import pandas as pd
import numpy as np
import optuna  # --- NEW ---
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor, early_stopping

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
print("Starting feature engineering...")

# Separate target variable and log-transform it
y = train_df["Transport_Cost"]
y_log = np.log1p(y)
train_df = train_df.drop(columns=["Transport_Cost", "Hospital_Id"])

# Store test IDs for submission
test_ids = test_df["Hospital_Id"]
test_df = test_df.drop(columns=["Hospital_Id"])

# Combine for consistent processing
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# --- Date Features (Expanded) ---
combined["Order_Placed_Date"] = pd.to_datetime(combined["Order_Placed_Date"], errors="coerce")
combined["Delivery_Date"] = pd.to_datetime(combined["Delivery_Date"], errors="coerce")

combined["Delivery_Duration_Days"] = (combined["Delivery_Date"] - combined["Order_Placed_Date"]).dt.days
combined["Order_Month"] = combined["Order_Placed_Date"].dt.month
combined["Order_DayOfWeek"] = combined["Order_Placed_Date"].dt.dayofweek
combined["Order_DayOfMonth"] = combined["Order_Placed_Date"].dt.dayofmonth # --- NEW ---
combined["Order_Quarter"] = combined["Order_Placed_Date"].dt.quarter     # --- NEW ---
combined["Order_Is_Weekend"] = (combined["Order_Placed_Date"].dt.dayofweek >= 5).astype(int)
combined["Order_Is_Month_End"] = combined["Order_Placed_Date"].dt.is_month_end.astype(int) # --- NEW ---

combined = combined.drop(columns=["Order_Placed_Date", "Delivery_Date"])

# --- Frequency Encoding ---
# How often does this customer or location appear?
cat_cols_for_freq = ["Customer_Id", "Location", "Pharmaceutical_Type"]
for col in cat_cols_for_freq:
    if col in combined.columns:
        freq = combined[col].value_counts()
        combined[f"{col}_freq"] = combined[col].map(freq)

# --- Interaction Features ---
# Combine two features to create a new one
if "Location" in combined.columns and "Pharmaceutical_Type" in combined.columns:
    combined["Location_Pharma_Interaction"] = combined["Location"].astype(str) + "_" + combined["Pharmaceutical_Type"].astype(str)

# --- Label Encoding (for all remaining objects) ---
cat_cols = combined.select_dtypes(include=["object"]).columns
for col in cat_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

# Define all categorical features for LGBM
lgbm_cat_features = cat_cols.tolist() + [
    "Order_Month", "Order_DayOfWeek", "Order_DayOfMonth",
    "Order_Quarter", "Order_Is_Weekend", "Order_Is_Month_End"
]
# Ensure all features are in the dataframe
lgbm_cat_features = [f for f in lgbm_cat_features if f in combined.columns]

# Split back into train/test
X_processed = combined.iloc[:len(train_df), :]
X_test_processed = combined.iloc[len(train_df):, :]

print(f"Feature engineering complete. Total features: {len(X_processed.columns)}")

# ---------------------------------------------------------------------------
# 3. OPTUNA HYPERPARAMETER TUNING
# ---------------------------------------------------------------------------

# We pass the data to the objective function
def objective(trial, X, y_log, y_orig):
    
    # --- Define the Hyperparameter Search Space ---
    params = {
        'objective': 'regression_l1',  # MAE
        'metric': 'mae',
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', -1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'boosting_type': 'gbdt',
    }
    
    # --- Run K-Fold Validation ---
    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_maes = []

    for fold, (train_index, val_index) in enumerate(kf.split(X, y_log)):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y_log.iloc[train_index], y_log.iloc[val_index]
        y_val_fold_orig = y_orig.iloc[val_index] # For final MAE calculation

        model = LGBMRegressor(**params)
        
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  eval_metric='mae',
                  callbacks=[early_stopping(100, verbose=False)],
                  categorical_feature=lgbm_cat_features)

        val_preds_log = model.predict(X_val_fold)
        val_preds = np.expm1(val_preds_log) # Convert back from log
        
        fold_mae = mean_absolute_error(y_val_fold_orig, val_preds)
        oof_maes.append(fold_mae)

    # Return the average MAE across all folds
    return np.mean(oof_maes)

# --- Create and run the Optuna study ---
print("Starting Optuna hyperparameter search...")
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X_processed, y_log, y), n_trials=50) # You can increase n_trials for a better search

print(f"Optuna search finished. Best MAE: {study.best_value}")
print("Best parameters found:")
print(study.best_params)

# ---------------------------------------------------------------------------
# 4. FINAL MODEL TRAINING (using best params)
# ---------------------------------------------------------------------------
print("Training final model using best parameters...")

# Get best params from the study
best_params = study.best_params
# Add back fixed parameters
best_params['n_estimators'] = 2000 # Use high n_estimators, early stopping will handle it
best_params['objective'] = 'regression_l1'
best_params['metric'] = 'mae'
best_params['random_state'] = 42
best_params['n_jobs'] = -1

N_SPLITS = 5 # Use the same fold strategy
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

test_predictions_list = []
oof_maes_final = []

for fold, (train_index, val_index) in enumerate(kf.split(X_processed, y_log)):
    print(f"--- Final Fold {fold+1}/{N_SPLITS} ---")
    X_train_fold, X_val_fold = X_processed.iloc[train_index], X_processed.iloc[val_index]
    y_train_fold, y_val_fold = y_log.iloc[train_index], y_log.iloc[val_index]
    y_val_fold_orig = y.iloc[val_index]

    model = LGBMRegressor(**best_params)
    
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              eval_metric='mae',
              callbacks=[early_stopping(100, verbose=True)],
              categorical_feature=lgbm_cat_features)

    val_preds_log = model.predict(X_val_fold)
    val_preds = np.expm1(val_preds_log)
    
    fold_mae = mean_absolute_error(y_val_fold_orig, val_preds)
    print(f"Fold {fold+1} MAE: {fold_mae:.2f}")
    oof_maes_final.append(fold_mae)
    
    # Predict on test data
    test_preds_log_fold = model.predict(X_test_processed)
    test_predictions_list.append(np.expm1(test_preds_log_fold))

print("---" * 10)
print(f"Overall OOF MAE (final model): {np.mean(oof_maes_final):.2f}")

# Average test predictions
final_test_predictions = np.mean(test_predictions_list, axis=0)

# ---------------------------------------------------------------------------
# 5. CREATE SUBMISSION
# ---------------------------------------------------------------------------
submission = pd.DataFrame({
    "Hospital_Id": test_ids,
    "Transport_Cost": final_test_predictions
})

# ✅ Safety checks
assert submission.shape == (500, 2), f"Submission shape incorrect: {submission.shape}"
assert set(submission["Hospital_Id"]) == set(test_df_orig["Hospital_Id"]), "❌ Hospital_Id values do not match test file!"
assert submission.columns.tolist() == ["Hospital_Id", "Transport_Cost"], "❌ Column names must be exactly: Hospital_Id, Transport_Cost"
print("Submission Hospital_Id count:", submission["Hospital_Id"].nunique())
print("Test Hospital_Id count:", test_df_orig["Hospital_Id"].nunique())
print("Any mismatch:", set(test_df_orig["Hospital_Id"]) - set(submission["Hospital_Id"]))

output_path = "submission_lgbm_tuned.csv"
submission.to_csv(output_path, index=False)
print(f"✅ {output_path} saved successfully!")
print(submission.head())