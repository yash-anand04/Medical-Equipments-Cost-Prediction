import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV  # Using LassoCV for faster/better tuning
from sklearn.model_selection import KFold # Using KFold for more robust validation
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================
# Step 1: Load datasets
# ============================================================
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print("Original train shape:", train_df.shape)
print("Original test shape:", test_df.shape)

# Save test Hospital_Id for submission
test_ids_for_submission = test_df['Hospital_Id']

# ============================================================
# Step 2: Fix negative Transport_Cost
# (Using your successful logic)
# ============================================================
train_df.loc[train_df['Transport_Cost'] < 0, 'Transport_Cost'] = 0
mean_positive_cost = train_df.loc[train_df['Transport_Cost'] > 0, 'Transport_Cost'].mean()
train_df.loc[train_df['Transport_Cost'] == 0, 'Transport_Cost'] = mean_positive_cost

# ============================================================
# Step 3: Process dates (Your swap logic + Our FE)
# ============================================================
def advanced_date_processing(df):
    df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], errors='coerce')
    df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')

    # Your brilliant swap logic
    invalid_mask = df['Order_Placed_Date'] > df['Delivery_Date']
    df.loc[invalid_mask, ['Order_Placed_Date', 'Delivery_Date']] = \
        df.loc[invalid_mask, ['Delivery_Date', 'Order_Placed_Date']].values

    # Delivery Duration
    df['Delivery_Duration'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days
    
    # --- ADDING OUR ADVANCED FEATURES ---
    df['Order_Month'] = df['Order_Placed_Date'].dt.month
    df['Order_DayOfWeek'] = df['Order_Placed_Date'].dt.dayofweek

    # Cyclical Features
    df['Order_Month_sin'] = np.sin(2 * np.pi * df['Order_Month']/12)
    df['Order_Month_cos'] = np.cos(2 * np.pi * df['Order_Month']/12)
    df['Order_DayOfWeek_sin'] = np.sin(2 * np.pi * df['Order_DayOfWeek']/7)
    df['Order_DayOfWeek_cos'] = np.cos(2 * np.pi * df['Order_DayOfWeek']/7)
    df["Order_Is_Weekend"] = (df['Order_DayOfWeek'] >= 5).astype(int)
    
    df = df.drop(columns=['Order_Placed_Date', 'Delivery_Date'])
    return df

train_df = advanced_date_processing(train_df)
test_df = advanced_date_processing(test_df)

# Impute Delivery_Duration (your logic)
mean_duration = train_df.loc[train_df['Delivery_Duration'] > 0, 'Delivery_Duration'].mean()
train_df.loc[train_df['Delivery_Duration'] <= 0, 'Delivery_Duration'] = mean_duration
train_df['Delivery_Duration'].fillna(mean_duration, inplace=True)
test_df.loc[test_df['Delivery_Duration'] <= 0, 'Delivery_Duration'] = mean_duration
test_df['Delivery_Duration'].fillna(mean_duration, inplace=True)

# ============================================================
# Step 4: Add Aggregation Features (Our Logic)
# ============================================================
# We do this *before* splitting to apply stats to test set
print("Adding aggregation features...")
full_df = pd.concat([train_df.drop(columns=['Transport_Cost']), test_df], axis=0)

hospital_agg_feats = full_df.groupby('Hospital_Id')['Delivery_Duration'].agg(['mean', 'std']).reset_index()
hospital_agg_feats.columns = ['Hospital_Id', 'Hospital_Dur_Mean', 'Hospital_Dur_Std']
hospital_order_count = full_df['Hospital_Id'].value_counts().reset_index()
hospital_order_count.columns = ['Hospital_Id', 'Hospital_Order_Count']

# Merge new features back
train_df = train_df.merge(hospital_agg_feats, on='Hospital_Id', how='left')
train_df = train_df.merge(hospital_order_count, on='Hospital_Id', how='left')
test_df = test_df.merge(hospital_agg_feats, on='Hospital_Id', how='left')
test_df = test_df.merge(hospital_order_count, on='Hospital_Id', how='left')

# Fill NaNs created by aggregation (e.g., std for single-order hospitals)
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# ============================================================
# Step 5: Split features and target
# ============================================================
X = train_df.drop(columns=['Transport_Cost'])
y = train_df['Transport_Cost']

# Identify all columns
# All our new features are numeric
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Found {len(num_cols)} numeric features.")
print(f"Found {len(cat_cols)} categorical features.")

# ============================================================
# Step 6: Preprocessing Pipeline (Your Framework)
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Set sparse=False for Lasso
        ]), cat_cols)
    ], remainder='passthrough') # Keep any columns we missed

# --- MODIFIED: Use LassoCV for automatic hyperparameter tuning ---
# LassoCV is much faster and more effective than GridSearchCV for this
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LassoCV(cv=5, n_jobs=-1, random_state=42, max_iter=2000, 
                          alphas=np.logspace(-4, 2, 100))) # Search 100 alphas
])

# ============================================================
# Step 7: Train Model
# ============================================================
print("Training model with LassoCV to find best alpha...")
# We train on the FULL dataset, as LassoCV has built-in cross-validation
pipeline.fit(X, y)

print(f"Best Alpha (from LassoCV): {pipeline.named_steps['regressor'].alpha_:.6f}")

# ============================================================
# Step 8: Evaluate (Optional, using K-Fold)
# This is just to get a reliable score, model is already trained
# ============================================================
print("Running 10-fold cross-validation for a reliable MAE score...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
val_maes = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # We use the *full* pipeline (preprocessor + model)
    # We fit the preprocessor on this fold's train data
    X_train_processed = pipeline.named_steps['preprocessor'].fit_transform(X_train)
    X_val_processed = pipeline.named_steps['preprocessor'].transform(X_val)
    
    # We train a new Lasso model *using the best alpha we already found*
    model = Lasso(alpha=pipeline.named_steps['regressor'].alpha_, max_iter=2000, random_state=42)
    model.fit(X_train_processed, y_train)
    
    val_preds = model.predict(X_val_processed)
    fold_mae = mean_absolute_error(y_val, val_preds)
    print(f"Fold {fold+1}/10 MAE: {fold_mae:.2f}")
    val_maes.append(fold_mae)

print(f"\nOverall OOF MAE: {np.mean(val_maes):.2f} +/- {np.std(val_maes):.2f}")

# ============================================================
# Step 9: Predict on Test Data
# ============================================================
print("Predicting on test data...")
test_predictions = pipeline.predict(test_df)

# Post-processing (your logic)
mean_positive_pred = test_predictions[test_predictions > 0].mean()
test_predictions = np.where(test_predictions < 0, mean_positive_pred, test_predictions)

# ============================================================
# Step 10: Create Submission File
# ============================================================
submission = pd.DataFrame({
    'Hospital_Id': test_ids_for_submission, # Use original IDs
    'Transport_Cost': test_predictions
})
submission.to_csv('submission_lasso_with_fe.csv', index=False)
print("\nSubmission file 'submission_lasso_with_fe.csv' created successfully!")
print(submission.head())