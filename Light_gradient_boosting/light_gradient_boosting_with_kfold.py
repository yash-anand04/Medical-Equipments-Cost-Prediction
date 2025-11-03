# transport_cost_advanced.py

import pandas as pd
import numpy as np  # --- NEW ---
from sklearn.model_selection import train_test_split, KFold  # --- MODIFIED ---
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor, early_stopping

# 1️⃣ Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2️⃣ Separate features and target
X = train.drop(columns=["Transport_Cost", "Hospital_Id"])
test_for_processing = test.drop(columns=["Hospital_Id"])

# --- MODIFIED ---
# We will log-transform the target variable 'y'
y = train["Transport_Cost"]
y_log = np.log1p(y)
# --- END MODIFICATION ---


# 3️⃣ Basic preprocessing
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

# --- LABEL ENCODING (Same as before) ---
cat_cols = combined.select_dtypes(include=["object"]).columns
for col in cat_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

lgbm_cat_features = cat_cols.tolist() + ["Order_Month", "Order_DayOfWeek", "Order_Is_Weekend"]
# --- END LABEL ENCODING ---

# Split back into train/test
X_processed = combined.iloc[:len(train), :]
X_test_processed = combined.iloc[len(train):, :]


# 4️⃣ & 5️⃣ --- MODIFIED: K-Fold Cross-Validation ---
print("Starting K-Fold Cross-Validation...")

N_SPLITS = 5  # You can change this (e.g., 10)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Arrays to store out-of-fold predictions and test predictions
oof_predictions = np.zeros(X_processed.shape[0])
test_predictions_list = []
oof_maes = []

for fold, (train_index, val_index) in enumerate(kf.split(X_processed, y_log)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train_fold, X_val_fold = X_processed.iloc[train_index], X_processed.iloc[val_index]
    y_train_fold, y_val_fold = y_log.iloc[train_index], y_log.iloc[val_index]
    y_val_fold_orig = y.iloc[val_index] # For calculating MAE on original scale

    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        # You can add more parameters here for tuning
        # num_leaves=31,
        # reg_alpha=0.1,
        # reg_lambda=0.1,
    )

    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              eval_metric='mae', # This MAE is on the log-scale, which is fine
              callbacks=[early_stopping(100, verbose=False)], # Set verbose=True to see more
              categorical_feature=lgbm_cat_features)

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
print(f"Overall OOF MAE: {np.mean(oof_maes):.2f}")

# 6️⃣ Average test predictions
# Average the predictions from all 5 folds
final_test_predictions = np.mean(test_predictions_list, axis=0)
# --- END MODIFICATION ---


# 7️⃣ Create submission file
submission = pd.DataFrame({
    "Hospital_Id": test["Hospital_Id"],
    "Transport_Cost": final_test_predictions # Use the averaged predictions
})

# ✅ Safety checks (same as before)
assert submission.shape == (500, 2), f"Submission shape incorrect: {submission.shape}"
assert set(submission["Hospital_Id"]) == set(test["Hospital_Id"]), "❌ Hospital_Id values do not match test file!"
assert submission.columns.tolist() == ["Hospital_Id", "Transport_Cost"], "❌ Column names must be exactly: Hospital_Id, Transport_Cost"
print("Submission Hospital_Id count:", submission["Hospital_Id"].nunique())
print("Test Hospital_Id count:", test["Hospital_Id"].nunique())
print("Any mismatch:", set(test["Hospital_Id"]) - set(submission["Hospital_Id"]))

# ✅ Save safely
output_path = "submission_lgbm_kfold_log.csv" # New name
submission.to_csv(output_path, index=False)
print(f"✅ {output_path} saved successfully!")
print(submission.shape)
print(submission.head())