# transport_cost_ensemble_final.py

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')

# 1️⃣ Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# --- ⭐️ NEW FIX HERE (v2) ⭐️ ---
# The error is from NEGATIVE costs. log1p(x) fails if x <= -1.
# We assume cost cannot be negative, so we "clip" all values < 0 to be 0.
negative_costs = train[train['Transport_Cost'] < 0]['Transport_Cost'].count()
if negative_costs > 0:
    print(f"Warning: Found {negative_costs} negative Transport_Cost values. Clipping them to 0.")
    train['Transport_Cost'] = train['Transport_Cost'].clip(lower=0)
else:
    print("No negative transport costs found. Checking for -inf values after log.")
# --- ⭐️ END FIX ⭐️ ---

# 2️⃣ Separate features and target
X = train.drop(columns=["Transport_Cost"]) 
test_for_processing = test.copy() 
test_ids_for_submission = test["Hospital_Id"]

# Log-transform the target (now safe from NaNs and -infs)
y = train["Transport_Cost"]
y_log = np.log1p(y)

# --- ⭐️ SANITY CHECK ⭐️ ---
# Check if any -inf values were created (e.g., from Transport_Cost = -1)
# This check is redundant if we clip at 0, but good to have.
inf_count = np.isinf(y_log).sum()
if inf_count > 0:
    print(f"Error: Found {inf_count} infinite values after log transform. Please check data for -1 values.")
    # Handle -inf if they exist, e.g., replace with 0
    y_log.replace([np.inf, -np.inf], 0, inplace=True) 
# --- ⭐️ END CHECK ⭐️ ---


# 3️⃣ Advanced Preprocessing & Feature Engineering
print("Starting advanced feature engineering...")
combined = pd.concat([X, test_for_processing], axis=0)

# --- DATE FEATURES ---
combined["Order_Placed_Date"] = pd.to_datetime(combined["Order_Placed_Date"], errors="coerce")
combined["Delivery_Date"] = pd.to_datetime(combined["Delivery_Date"], errors="coerce")
combined["Order_Month"] = combined["Order_Placed_Date"].dt.month
combined["Order_DayOfWeek"] = combined["Order_Placed_Date"].dt.dayofweek
combined["Order_Is_Weekend"] = (combined["Order_Placed_Date"].dt.dayofweek >= 5).astype(int)
combined["Delivery_Duration_Days"] = (combined["Delivery_Date"] - combined["Order_Placed_Date"]).dt.days
combined.drop(columns=["Order_Placed_Date", "Delivery_Date"], inplace=True)

# --- IMPUTATION (Numerical) ---
duration_median = combined['Delivery_Duration_Days'].median()
combined['Delivery_Duration_Days'] = combined['Delivery_Duration_Days'].fillna(duration_median)

# --- IMPUTATION (Categorical) ---
cat_cols_raw = combined.select_dtypes(include=["object", "category"]).columns
for col in cat_cols_raw:
    combined[col] = combined[col].fillna("Missing")

# --- AGGREGATION FEATURES ---
hospital_agg_feats = combined.groupby('Hospital_Id')['Delivery_Duration_Days'].agg(['mean', 'std', 'max', 'min']).reset_index()
hospital_agg_feats.columns = ['Hospital_Id', 'Hospital_Dur_Mean', 'Hospital_Dur_Std', 'Hospital_Dur_Max', 'Hospital_Dur_Min']
hospital_order_count = combined['Hospital_Id'].value_counts().reset_index()
hospital_order_count.columns = ['Hospital_Id', 'Hospital_Order_Count']

combined = combined.merge(hospital_agg_feats, on='Hospital_Id', how='left')
combined = combined.merge(hospital_order_count, on='Hospital_Id', how='left')

combined['Hospital_Dur_Std'] = combined['Hospital_Dur_Std'].fillna(0)

# --- INTERACTION FEATURES (Numerical) ---
print("Creating numerical interaction features...")
combined['Duration_vs_Hospital_Mean'] = combined['Delivery_Duration_Days'] / combined['Hospital_Dur_Mean']
combined['Duration_vs_Hospital_Max'] = combined['Delivery_Duration_Days'] - combined['Hospital_Dur_Max']

# Fill any NaNs/Infs created by division
combined.replace([np.inf, -np.inf], np.nan, inplace=True)
num_cols = combined.select_dtypes(include=np.number).columns
for col in num_cols:
    if combined[col].isnull().any():
        median_val = combined[col].median()
        combined[col] = combined[col].fillna(median_val)
# --- END FE ---


# --- 4️⃣ Prepare Data for TWO Models ---
# Using the list from your log
cat_features_list = ['Hospital_Id', 'Supplier_Name', 'Equipment_Type', 'CrossBorder_Shipping', 
                     'Urgent_Shipping', 'Installation_Service', 'Transport_Method', 
                     'Fragile_Equipment', 'Hospital_Info', 'Rural_Hospital', 'Hospital_Location', 
                     'Order_Month', 'Order_DayOfWeek', 'Order_Is_Weekend']

cat_features_list = [col for col in cat_features_list if col in combined.columns]
print(f"Found {len(cat_features_list)} categorical features: {cat_features_list}")


# --- Prep for CatBoost (Strings are OK) ---
X_processed_catboost = combined.iloc[:len(train), :].copy()
X_test_processed_catboost = combined.iloc[len(train):, :].copy()

for col in cat_features_list:
    X_processed_catboost[col] = X_processed_catboost[col].astype(str)
    X_test_processed_catboost[col] = X_test_processed_catboost[col].astype(str)


# --- Prep for LightGBM (Requires Label Encoding) ---
print("Label Encoding for LightGBM...")
combined_lgbm = combined.copy()
lgbm_cat_features = [] 

for col in cat_features_list:
    if col in combined_lgbm.columns:
        combined_lgbm[col] = combined_lgbm[col].astype(str)
        le = LabelEncoder()
        combined_lgbm[col] = le.fit_transform(combined_lgbm[col])
        lgbm_cat_features.append(col) 

X_processed_lgbm = combined_lgbm.iloc[:len(train), :]
X_test_processed_lgbm = combined_lgbm.iloc[len(train):, :]
# --- End Data Prep ---


# 5️⃣ K-Fold Cross-Validation with ENSEMBLE
print("Starting K-Fold Ensemble Training...")

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

test_preds_lgbm_list = []
test_preds_catboost_list = []
oof_maes_lgbm = []
oof_maes_catboost = []

for fold, (train_index, val_index) in enumerate(kf.split(X_processed_lgbm, y_log)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    y_train_fold, y_val_fold = y_log.iloc[train_index], y_log.iloc[val_index]
    y_val_fold_orig = y.iloc[val_index]

    # --- 1. LightGBM ---
    print("Training LGBM...")
    X_train_lgbm, X_val_lgbm = X_processed_lgbm.iloc[train_index], X_processed_lgbm.iloc[val_index]
    
    model_lgbm = LGBMRegressor(
        n_estimators=5000, learning_rate=0.01, num_leaves=64,
        reg_alpha=0.1, reg_lambda=0.1, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    
    model_lgbm.fit(X_train_lgbm, y_train_fold,
                   eval_set=[(X_val_lgbm, y_val_fold)],
                   eval_metric='mae',
                   callbacks=[early_stopping(200, verbose=False)],
                   categorical_feature=lgbm_cat_features) 
    
    val_preds_lgbm = np.expm1(model_lgbm.predict(X_val_lgbm))
    fold_mae_lgbm = mean_absolute_error(y_val_fold_orig, val_preds_lgbm)
    print(f"LGBM Fold {fold+1} MAE: {fold_mae_lgbm:.2f}")
    oof_maes_lgbm.append(fold_mae_lgbm)
    
    test_preds_lgbm_fold = np.expm1(model_lgbm.predict(X_test_processed_lgbm))
    test_preds_lgbm_list.append(test_preds_lgbm_fold)

    # --- 2. CatBoost ---
    print("Training CatBoost...")
    X_train_cat, X_val_cat = X_processed_catboost.iloc[train_index], X_processed_catboost.iloc[val_index]

    model_catboost = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.01,
        eval_metric='MAE',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=200,
        l2_leaf_reg=3,
    )

    model_catboost.fit(X_train_cat, y_train_fold,
                       eval_set=[(X_val_cat, y_val_fold)],
                       cat_features=cat_features_list, 
                       use_best_model=True
                      )
    
    val_preds_catboost = np.expm1(model_catboost.predict(X_val_cat))
    fold_mae_catboost = mean_absolute_error(y_val_fold_orig, val_preds_catboost)
    print(f"CatBoost Fold {fold+1} MAE: {fold_mae_catboost:.2f}")
    oof_maes_catboost.append(fold_mae_catboost)

    test_preds_catboost_fold = np.expm1(model_catboost.predict(X_test_processed_catboost))
    test_preds_catboost_list.append(test_preds_catboost_fold)

print("---" * 15)
print(f"Overall OOF LGBM MAE: {np.mean(oof_maes_lgbm):.2f} +/- {np.std(oof_maes_lgbm):.2f}")
print(f"Overall OOF CatBoost MAE: {np.mean(oof_maes_catboost):.2f} +/- {np.std(oof_maes_catboost):.2f}")

# 6️⃣ Average test predictions (ENSEMBLE)
print("Averaging ensemble predictions...")
final_test_preds_lgbm = np.mean(test_preds_lgbm_list, axis=0)
final_test_preds_catboost = np.mean(test_preds_catboost_list, axis=0)

# Simple 50/50 Average Ensemble
final_test_predictions = (final_test_preds_lgbm * 0.5) + (final_test_preds_catboost * 0.5)


# 7️⃣ Create submission file
submission = pd.DataFrame({
    "Hospital_Id": test_ids_for_submission,
    "Transport_Cost": final_test_predictions
})

# ✅ Safety checks
assert submission.shape == (500, 2), f"Submission shape incorrect: {submission.shape}"
assert set(submission["Hospital_Id"]) == set(test_ids_for_submission), "❌ Hospital_Id values do not match test file!"
assert submission.columns.tolist() == ["Hospital_Id", "Transport_Cost"], "❌ Column names must be exactly: Hospital_Id, Transport_Cost"

# ✅ Save safely
output_path = "submission_ensemble_final_v2.csv" # New name
submission.to_csv(output_path, index=False)
print(f"✅ {output_path} saved successfully!")
print(submission.shape)
print(submission.head())