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
# ============================================================
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
# Step 3: !!! NEW - OUTLIER REMOVAL !!!
# This is the key to fixing the 47,000 MAE folds.
# We remove the extreme top-end costs from the *training* data.
# ============================================================
PERCENTILE_CLIP = 0.995 # Remove top 0.5%
clip_value = train_df['Transport_Cost'].quantile(PERCENTILE_CLIP)

print(f"Original training data shape: {train_df.shape}")
print(f"Clipping training data at {PERCENTILE_CLIP*100}th percentile: Transport_Cost > {clip_value:.2f}")

# Keep only the "normal" data for training
train_df = train_df[train_df['Transport_Cost'] < clip_value]
print(f"New training data shape: {train_df.shape}")
# ============================================================

# ============================================================
# Step 4: Process dates (Your Logic + Light FE)
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

# We process the *new, smaller* train_df and the *full* test_df
train_df = process_dates_with_swap(train_df)
test_df = process_dates_with_swap(test_df)

# ============================================================
# Step 5: Split features and target
# ============================================================
X = train_df.drop(columns=['Transport_Cost'])
y = train_df['Transport_Cost'] # This is now the *clean* target

# --- Explicitly define ALL columns to avoid dtype ambiguity ---
cat_cols = [
    'Hospital_Id', 'Supplier_Name', 'Equipment_Type', 'CrossBorder_Shipping', 
    'Urgent_Shipping', 'Installation_Service', 'Transport_Method', 
    'Fragile_Equipment', 'Hospital_Info', 'Rural_Hospital', 'Hospital_Location'
]
num_cols = [
    'Order_Year', 'Patient_Age', 'Patient_Blood_Type', 
    'Distance_km', 'Weight_kg', 'Storage_Temperature_C', 
    'Delivery_Duration', 'Order_Month', 'Order_DayOfWeek', 'Order_Is_Weekend'
]
cat_cols = [col for col in cat_cols if col in X.columns]
num_cols = [col for col in num_cols if col in X.columns]

print(f"Using {len(num_cols)} numeric features.")
print(f"Using {len(cat_cols)} categorical features.")

# Force types to be consistent in both train (X) and test
X[cat_cols] = X[cat_cols].astype(str)
X[num_cols] = X[num_cols].astype(float)

test_df[cat_cols] = test_df[cat_cols].astype(str)
test_df[num_cols] = test_df[num_cols].astype(float)
# ============================================================

# ============================================================
# Step 6: Preprocessing (Your Pipeline)
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')), 
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

# Use LassoCV to find the best alpha
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LassoCV(cv=5, n_jobs=-1, random_state=42, max_iter=2000, 
                          alphas=np.logspace(-3, 2, 50)))
])

# ============================================================
# Step 7: Train Model on Full (Cleaned) Data
# ============================================================
print("\nTraining model with LassoCV on *cleaned* full dataset...")
# X and y are now the cleaned versions
pipeline.fit(X, y)

best_alpha = pipeline.named_steps['regressor'].alpha_
print(f"Best Alpha (from LassoCV): {best_alpha}")

# ============================================================
# Step 8: Evaluate on Validation Set (More Robustly)
# ============================================================
print("\nRunning 10-fold cross-validation on *cleaned* data...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
val_maes = []

final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=best_alpha, max_iter=2000, random_state=42))
])

# X and y are our cleaned data
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    final_pipeline.fit(X_train, y_train)
    val_preds = final_pipeline.predict(X_val)
    fold_mae = mean_absolute_error(y_val, val_preds)
    print(f"Fold {fold+1}/10 MAE: {fold_mae:.2f}")
    val_maes.append(fold_mae)

# This MAE should be MUCH lower and the +/- std dev should be tiny
print(f"\nOverall OOF MAE (Cleaned): {np.mean(val_maes):.2f} +/- {np.std(val_maes):.2f}")

# ============================================================
# Step 9: Predict on Test Data
# ============================================================
print("Predicting on *original* test data...")
# We use the pipeline trained on clean data to predict on the real test data
test_predictions = pipeline.predict(test_df)

# Post-processing (Your logic)
mean_positive_pred = test_predictions[test_predictions > 0].mean()
test_predictions = np.where(test_predictions < 0, mean_positive_pred, test_predictions)

# ============================================================
# Step 10: Create Submission File
# ============================================================
submission = pd.DataFrame({
    'Hospital_Id': test_ids_for_submission,
    'Transport_Cost': test_predictions
})
submission.to_csv('submission_lasso_v4_outliers_removed.csv', index=False)
print("\nSubmission file 'submission_lasso_v4_outliers_removed.csv' created successfully!")
print(submission.head())