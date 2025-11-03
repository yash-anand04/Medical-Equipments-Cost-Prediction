# transport_cost_improved.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor, early_stopping  # --- NEW ---

# 1️⃣ Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2️⃣ Separate features and target
# --- MODIFIED ---
# Drop Hospital_Id *before* combining. It's an ID, not a feature.
X = train.drop(columns=["Transport_Cost", "Hospital_Id"])
test_for_processing = test.drop(columns=["Hospital_Id"])
# --- END MODIFICATION ---

y = train["Transport_Cost"]

# 3️⃣ Basic preprocessing
# --- MODIFIED ---
# Combine train & test for consistent encoding
combined = pd.concat([X, test_for_processing], axis=0)
# --- END MODIFICATION ---

# --- FIXED DATE PROCESSING ---
combined["Order_Placed_Date"] = pd.to_datetime(combined["Order_Placed_Date"], errors="coerce")
combined["Delivery_Date"] = pd.to_datetime(combined["Delivery_Date"], errors="coerce")

# --- NEW FEATURES ---
combined["Order_Month"] = combined["Order_Placed_Date"].dt.month
combined["Order_DayOfWeek"] = combined["Order_Placed_Date"].dt.dayofweek
combined["Order_Is_Weekend"] = (combined["Order_Placed_Date"].dt.dayofweek >= 5).astype(int)
# --- END NEW FEATURES ---

combined["Delivery_Duration_Days"] = (combined["Delivery_Date"] - combined["Order_Placed_Date"]).dt.days

combined.drop(columns=["Order_Placed_Date", "Delivery_Date"], inplace=True)
# --- END FIX ---

# Label encode categorical columns
cat_cols = combined.select_dtypes(include=["object"]).columns
for col in cat_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

# --- NEW ---
# Define categorical features for LGBM
# This is the list of original object columns + our new date-based integers
lgbm_cat_features = cat_cols.tolist() + ["Order_Month", "Order_DayOfWeek", "Order_Is_Weekend"]
# --- END NEW ---

# Split back into train/test
X_processed = combined.iloc[:len(train), :]
X_test_processed = combined.iloc[len(train):, :]

# 4️⃣ Train/validation split for quick check
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 5️⃣ Train a powerful LGBM model
# --- MODIFIED ---
model = LGBMRegressor(
    n_estimators=2000,  # Increased, but early stopping will find the best number
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1,
    # boosting_type='gbdt' # default
)

print("Training with LGBM...")
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='mae',
          callbacks=[early_stopping(100, verbose=True)],  # Stop after 100 rounds if no improvement
          categorical_feature=lgbm_cat_features)
# --- END MODIFICATION ---

# Evaluate
y_pred_val = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred_val)
print(f"Validation MAE (LGBM): {mae:.2f}")

# 6️⃣ Predict on test set
test_predictions = model.predict(X_test_processed)

# 7️⃣ Create submission file with correct headers and IDs
submission = pd.DataFrame({
    "Hospital_Id": test["Hospital_Id"],    # use original test IDs
    "Transport_Cost": test_predictions     # your predicted target
})

# ✅ Safety checks before saving
assert submission.shape == (500, 2), f"Submission shape incorrect: {submission.shape}"
assert set(submission["Hospital_Id"]) == set(test["Hospital_Id"]), "❌ Hospital_Id values do not match test file!"
assert submission.columns.tolist() == ["Hospital_Id", "Transport_Cost"], "❌ Column names must be exactly: Hospital_Id, Transport_Cost"
print("Submission Hospital_Id count:", submission["Hospital_Id"].nunique())
print("Test Hospital_Id count:", test["Hospital_Id"].nunique())
print("Any mismatch:", set(test["Hospital_Id"]) - set(submission["Hospital_Id"]))

# ✅ Save safely
output_path = "submission_lgbm.csv" # New name
submission.to_csv(output_path, index=False)
print(f"✅ {output_path} saved successfully!")
print(submission.shape)
print(submission.head())