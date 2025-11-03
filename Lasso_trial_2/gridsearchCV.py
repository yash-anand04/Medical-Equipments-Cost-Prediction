import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LassoCV # --- MODIFIED ---
from sklearn.model_selection import KFold # --- MODIFIED ---
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
# Step 3: Process dates (Your Logic + Light FE)
# ============================================================
def process_dates_with_swap(df):
    df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], errors='coerce')
    df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')

    invalid_mask = df['Order_Placed_Date'] > df['Delivery_Date']
    df.loc[invalid_mask, ['Order_Placed_Date', 'Delivery_Date']] = \
        df.loc[invalid_mask, ['Delivery_Date', 'Order_Placed_Date']].values

    df['Delivery_Duration'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days
    df['Delivery_Duration'] = df['Delivery_Duration'].astype(float) # Avoid dtype warnings

    # --- REITERATION 1: Add simple date features ---
    df['Order_Month'] = df['Order_Placed_Date'].dt.month
    df['Order_DayOfWeek'] = df['Order_Placed_Date'].dt.dayofweek
    df["Order_Is_Weekend"] = (df['Order_DayOfWeek'] >= 5).astype(int)
    # --- END REITERATION ---

    mean_duration = df.loc[df['Delivery_Duration'] > 0, 'Delivery_Duration'].mean()
    df.loc[df['Delivery_Duration'] <= 0, 'Delivery_Duration'] = mean_duration
    df['Delivery_Duration'] = df['Delivery_Duration'].fillna(mean_duration)
    
    # Drop original date columns
    df = df.drop(columns=['Order_Placed_Date', 'Delivery_Date'])
    return df

train_df = process_dates_with_swap(train_df)
test_df = process_dates_with_swap(test_df)

# ============================================================
# Step 4: Split features and target
# ============================================================
X = train_df.drop(columns=['Transport_Cost'])
y = train_df['Transport_Cost']

# --- MODIFIED: Automatically find all num/cat columns ---
# Our new date features will be automatically included in num_cols
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Found {len(num_cols)} numeric features: {num_cols}")
print(f"Found {len(cat_cols)} categorical features: {cat_cols}")

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
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

# --- REITERATION 2: Use LassoCV for better/faster tuning ---
# This replaces GridSearchCV and trains on the *full* dataset
# It finds the best alpha using its own internal cross-validation
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LassoCV(cv=5, n_jobs=-1, random_state=42, max_iter=2000, 
                          alphas=np.logspace(-3, 2, 50))) # Search 50 alphas
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
# Now we run our own K-Fold to get a reliable OOF score
print("\nRunning 10-fold cross-validation for a reliable MAE score...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
val_maes = []

# We create a new pipeline with the *best alpha* we just found
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
# Use the main pipeline that was trained on the FULL dataset
test_predictions = pipeline.predict(test_df)

# Post-processing (Your logic)
mean_positive_pred = test_predictions[test_predictions > 0].mean()
test_predictions = np.where(test_predictions < 0, mean_positive_pred, test_predictions)

# ============================================================
# Step 9: Create Submission File
# ============================================================
# --- MODIFIED: Fixed column names to match Kaggle ---
submission = pd.DataFrame({
    'Hospital_Id': test_ids_for_submission,
    'Transport_Cost': test_predictions
})
submission.to_csv('submission_lasso_v2.csv', index=False)
print("\nSubmission file 'submission_lasso_v2.csv' created successfully!")
print(submission.head())