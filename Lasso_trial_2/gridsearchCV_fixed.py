import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Suppress future warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# ============================================================
# Step 1: Load datasets
# =================================_===========
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Save test Hospital_Id for submission
test_ids_for_submission = test_df['Hospital_Id']

# ============================================================
# Step 2: Fix negative Transport_Cost (Your Logic)
# ============================================================
train_df.loc[train_df['Transport_Cost'] < 0, 'Transport_Cost'] = 0
mean_positive_cost = train_df.loc[train_df['Transport_Cost'] > 0, 'Transport_Cost'].mean()
train_df.loc[train_df['Transport_Cost'] == 0, 'Transport_Cost'] = mean_positive_cost

# ============================================================
# Step 3: Process dates (Your Logic + Light FE)
# ============================================================
def process_dates_with_swap(df):
    df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], errors='coerce')
    df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')

    invalid_mask = df['Order_Placed_Date'] > df['Delivery_Date']
    df.loc[invalid_mask, ['Order_Placed_Date', 'Delivery_Date']] = \
        df.loc[invalid_mask, ['Delivery_Date', 'Order_Placed_Date']].values

    df['Delivery_Duration'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days
    df['Delivery_Duration'] = df['Delivery_Duration'].astype(float) 

    df['Order_Month'] = df['Order_Placed_Date'].dt.month
    df['Order_DayOfWeek'] = df['Order_Placed_Date'].dt.dayofweek
    df["Order_Is_Weekend"] = (df['Order_DayOfWeek'] >= 5).astype(int)
    
    df = df.drop(columns=['Order_Placed_Date', 'Delivery_Date'])
    
    mean_duration = df.loc[df['Delivery_Duration'] > 0, 'Delivery_Duration'].mean()
    df.loc[df['Delivery_Duration'] <= 0, 'Delivery_Duration'] = mean_duration
    df['Delivery_Duration'] = df['Delivery_Duration'].fillna(mean_duration)
    
    return df

train_df = process_dates_with_swap(train_df)
test_df = process_dates_with_swap(test_df)

# ============================================================
# Step 4: Split features and target
# ============================================================
X = train_df.drop(columns=['Transport_Cost'])
y = train_df['Transport_Cost']

# --- ⭐️ HERE IS THE FIX ⭐️ ---
# Explicitly define ALL columns to avoid dtype ambiguity

# All columns from the file that are categorical (text)
cat_cols = [
    'Hospital_Id', 'Supplier_Name', 'Equipment_Type', 'CrossBorder_Shipping', 
    'Urgent_Shipping', 'Installation_Service', 'Transport_Method', 
    'Fragile_Equipment', 'Hospital_Info', 'Rural_Hospital', 'Hospital_Location'
]

# All columns that are numeric (including our new date features)
num_cols = [
    'Order_Year', 'Patient_Age', 'Patient_Blood_Type', 
    'Distance_km', 'Weight_kg', 'Storage_Temperature_C', 
    'Delivery_Duration', 'Order_Month', 'Order_DayOfWeek', 'Order_Is_Weekend'
]

# Ensure the columns actually exist in the dataframe
cat_cols = [col for col in cat_cols if col in X.columns]
num_cols = [col for col in num_cols if col in X.columns]

print(f"Using {len(num_cols)} numeric features.")
print(f"Using {len(cat_cols)} categorical features.")

# Force types to be consistent in both train (X) and test
X[cat_cols] = X[cat_cols].astype(str)
X[num_cols] = X[num_cols].astype(float)

test_df[cat_cols] = test_df[cat_cols].astype(str)
test_df[num_cols] = test_df[num_cols].astype(float)
# --- ⭐️ END FIX ⭐️ ---

# ============================================================
# Step 5: Preprocessing (Your Pipeline)
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            # Impute with "Missing" just in case we missed one
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')), 
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

# Use LassoCV for automatic hyperparameter tuning
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LassoCV(cv=5, n_jobs=-1, random_state=42, max_iter=2000, 
                          alphas=np.logspace(-3, 2, 50)))
])

# ============================================================
# Step 6: Train Model on Full Data
# ============================================================
print("\nTraining model with LassoCV on full dataset...")
pipeline.fit(X, y)

best_alpha = pipeline.named_steps['regressor'].alpha_
print(f"Best Alpha (from LassoCV): {best_alpha}")

# ============================================================
# Step 7: Evaluate on Validation Set (More Robustly)
# ============================================================
print("\nRunning 10-fold cross-validation for a reliable MAE score...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
val_maes = []

final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=best_alpha, max_iter=2000, random_state=42))
])

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    final_pipeline.fit(X_train, y_train)
    val_preds = final_pipeline.predict(X_val)
    fold_mae = mean_absolute_error(y_val, val_preds)
    print(f"Fold {fold+1}/10 MAE: {fold_mae:.2f}")
    val_maes.append(fold_mae)

print(f"\nOverall OOF MAE: {np.mean(val_maes):.2f} +/- {np.std(val_maes):.2f}")

# ============================================================
# Step 8: Predict on Test Data
# ============================================================
print("Predicting on test data...")
# This should now work!
test_predictions = pipeline.predict(test_df)

# Post-processing (Your logic)
mean_positive_pred = test_predictions[test_predictions > 0].mean()
test_predictions = np.where(test_predictions < 0, mean_positive_pred, test_predictions)

# =================================_==========================
# Step 9: Create Submission File
# ============================================================
submission = pd.DataFrame({
    'Hospital_Id': test_ids_for_submission,
    'Transport_Cost': test_predictions
})
submission.to_csv('submission_lasso_v3_fixed.csv', index=False)
print("\nSubmission file 'submission_lasso_v3_fixed.csv' created successfully!")
print(submission.head())