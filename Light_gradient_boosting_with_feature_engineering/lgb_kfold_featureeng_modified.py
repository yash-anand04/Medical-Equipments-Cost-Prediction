# transport_cost_final.py

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor, early_stopping
import warnings

warnings.filterwarnings('ignore')

# 1️⃣ Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2️⃣ Separate features and target
# --- MODIFIED: We keep 'Hospital_Id' in X now! ---
X = train.drop(columns=["Transport_Cost"]) 
test_for_processing = test.copy() # Keep all columns for processing
test_ids_for_submission = test["Hospital_Id"] # Save original IDs for submission file

# We will log-transform the target variable 'y'
y = train["Transport_Cost"]
y_log = np.log1p(y)
# --- END MODIFICATION ---


# 3️⃣ Advanced Preprocessing & Feature Engineering
combined = pd.concat([X, test_for_processing], axis=0)

# --- DATE FEATURES (Same as before) ---
combined["Order_Placed_Date"] = pd.to_datetime(combined["Order_Placed_Date"], errors="coerce")
combined["Delivery_Date"] = pd.to_datetime(combined["Delivery_Date"], errors="coerce")
combined["Order_Month"] = combined["Order_Placed_Date"].dt.month
combined["Order_DayOfWeek"] = combined["Order_Placed_Date"].dt.dayofweek
combined["Order_Is_Weekend"] = (combined["Order_Placed_Date"].dt.dayofweek >= 5).astype(int)
combined["Delivery_Duration_Days"] = (combined["Delivery_Date"] - combined["Order_Placed_Date"]).dt.days
combined.drop(columns=["Order_Placed_Date", "Delivery_Date"], inplace=True)
# --- END DATE FEATURES ---

# --- IMPUTATION (NEW) ---
# Fill NaNs for duration *before* aggregating. Using median is robust.
duration_median = combined['Delivery_Duration_Days'].median()
combined['Delivery_Duration_Days'] = combined['Delivery_Duration_Days'].fillna(duration_median)

# Impute other potential numerical NaNs (if any)
num_cols = combined.select_dtypes(include=np.number).columns
for col in num_cols:
    if combined[col].isnull().any():
        median_val = combined[col].median()
        combined[col] = combined[col].fillna(median_val)
# --- END IMPUTATION ---

# --- AGGREGATION FEATURES (NEW) ---
# Create powerful new features by grouping by Hospital_Id
print("Creating aggregation features...")

# Stats for Delivery_Duration_Days per hospital
hospital_agg_feats = combined.groupby('Hospital_Id')['Delivery_Duration_Days'].agg(['mean', 'std', 'max', 'min']).reset_index()
hospital_agg_feats.columns = ['Hospital_Id', 'Hospital_Dur_Mean', 'Hospital_Dur_Std', 'Hospital_Dur_Max', 'Hospital_Dur_Min']

# Count of orders per hospital
hospital_order_count = combined['Hospital_Id'].value_counts().reset_index()
hospital_order_count.columns = ['Hospital_Id', 'Hospital_Order_Count']

# Merge new features back
combined = combined.merge(hospital_agg_feats, on='Hospital_Id', how='left')
combined = combined.merge(hospital_order_count, on='Hospital_Id', how='left')

# Fill NaNs created by aggregation (e.g., std for single-order hospitals)
combined['Hospital_Dur_Std'] = combined['Hospital_Dur_Std'].fillna(0)
# --- END AGGREGATION FEATURES ---

# --- LABEL ENCODING (MODIFIED) ---
# We now identify all categorical features, including Hospital_Id and our new date features
cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()

# Manually add features we know are categorical
manual_cat_features = ["Hospital_Id", "Order_Month", "Order_DayOfWeek", "Order_Is_Weekend"]
for col in manual_cat_features:
    if col not in cat_cols:
        cat_cols.append(col)

print(f"Treating {len(cat_cols)} columns as categorical: {cat_cols}")
lgbm_cat_features = cat_cols.copy() # Save list for LGBM

# Apply LabelEncoder to all categorical columns
for col in lgbm_cat_features:
    # Convert to string first to handle mixed types or integers safely
    combined[col] = combined[col].astype(str)
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
# --- END LABEL ENCODING ---

# Split back into train/test
X_processed = combined.iloc[:len(train), :]
X_test_processed = combined.iloc[len(train):, :]

print(f"Processed training features shape: {X_processed.shape}")
print(f"Processed test features shape: {X_test_processed.shape}")

# 4️⃣ & 5️⃣ K-Fold Cross-Validation (MODIFIED Hyperparameters)
print("Starting K-Fold Cross-Validation...")

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_predictions = np.zeros(X_processed.shape[0])
test_predictions_list = []
oof_maes = []

for fold, (train_index, val_index) in enumerate(kf.split(X_processed, y_log)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train_fold, X_val_fold = X_processed.iloc[train_index], X_processed.iloc[val_index]
    y_train_fold, y_val_fold = y_log.iloc[train_index], y_log.iloc[val_index]
    y_val_fold_orig = y.iloc[val_index] # For calculating MAE on original scale

    # --- MODIFIED: Tuned Hyperparameters ---
    model = LGBMRegressor(
        n_estimators=5000,         # Increased, but early stopping will find the best
        learning_rate=0.01,        # Lowered for better accuracy
        num_leaves=64,             # Increased from default (31) for more complexity
        reg_alpha=0.1,             # L1 regularization
        reg_lambda=0.1,            # L2 regularization
        subsample=0.8,             # Use 80% of data for training each tree
        colsample_bytree=0.8,      # Use 80% of features for training each tree
        random_state=42,
        n_jobs=-1,
        boosting_type='gbdt',
    )

    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              eval_metric='mae', # MAE on the log-scale
              callbacks=[early_stopping(200, verbose=False)], # Increased patience
              categorical_feature=lgbm_cat_features)
    # --- END MODIFICATION ---

    # Predict on validation data (log scale)
    val_preds_log = model.predict(X_val_fold)
    
    # --- IMPORTANT: Convert predictions back from log scale ---
    val_preds = np.expm1(val_preds_log)
    
    # Store OOF preds
    oof_predictions[val_index] = val_preds
    
    # Calculate MAE on the *original* scale
    fold_mae = mean_absolute_error(y_val_fold_orig, val_preds)
    print(f"Fold {fold+1} MAE: {fold_mae:.2f}")
    oof_maes.append(fold_mae)
    
    # Predict on test data (log scale) and store
    test_preds_log_fold = model.predict(X_test_processed)
    test_predictions_list.append(np.expm1(test_preds_log_fold)) # Convert back

print("---" * 10)
print(f"Overall OOF MAE (Mean Absolute Error): {np.mean(oof_maes):.2f}")
print(f"Overall OOF MAE (Standard Deviation): {np.std(oof_maes):.2f}")

# 6️⃣ Average test predictions
final_test_predictions = np.mean(test_predictions_list, axis=0)


# 7️⃣ Create submission file
submission = pd.DataFrame({
    "Hospital_Id": test_ids_for_submission, # Use the original test IDs
    "Transport_Cost": final_test_predictions
})

# ✅ Safety checks (same as before)
assert submission.shape == (500, 2), f"Submission shape incorrect: {submission.shape}"
assert set(submission["Hospital_Id"]) == set(test_ids_for_submission), "❌ Hospital_Id values do not match test file!"
assert submission.columns.tolist() == ["Hospital_Id", "Transport_Cost"], "❌ Column names must be exactly: Hospital_Id, Transport_Cost"
print("Submission Hospital_Id count:", submission["Hospital_Id"].nunique())
print("Test Hospital_Id count:", test_ids_for_submission.nunique())
print("Any mismatch:", set(test_ids_for_submission) - set(submission["Hospital_Id"]))

# ✅ Save safely
output_path = "submission_lgbm_kfold_log_fe.csv" # New name for new features
submission.to_csv(output_path, index=False)
print(f"✅ {output_path} saved successfully!")
print(submission.shape)
print(submission.head())