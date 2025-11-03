import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler # --- NEW: RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge, ElasticNet # --- NEW: Added Ridge/ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ============================================================
# Step 1: Load datasets
# ============================================================
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Save original test Hospital_Id for submission
test_ids_for_submission = test_df['Hospital_Id']

# ============================================================
# Step 2: Fix negative Transport_Cost (Your Logic)
# ============================================================
train_df.loc[train_df['Transport_Cost'] < 0, 'Transport_Cost'] = 0
mean_positive_cost = train_df.loc[train_df['Transport_Cost'] > 0, 'Transport_Cost'].mean()
train_df.loc[train_df['Transport_Cost'] == 0, 'Transport_Cost'] = mean_positive_cost

# ============================================================
# Step 3: Process dates (Your Logic)
# ============================================================
def process_dates_with_swap(df):
    df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], errors='coerce')
    df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')
    invalid_mask = df['Order_Placed_Date'] > df['Delivery_Date']
    df.loc[invalid_mask, ['Order_Placed_Date', 'Delivery_Date']] = \
        df.loc[invalid_mask, ['Delivery_Date', 'Order_Placed_Date']].values
    df['Delivery_Duration'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days
    mean_duration = df.loc[df['Delivery_Duration'] > 0, 'Delivery_Duration'].mean()
    df.loc[df['Delivery_Duration'] <= 0, 'Delivery_Duration'] = mean_duration
    df['Delivery_Duration'].fillna(mean_duration, inplace=True)
    return df

train_df = process_dates_with_swap(train_df)
test_df = process_dates_with_swap(test_df)
train_df = train_df.drop(columns=['Order_Placed_Date', 'Delivery_Date'])
test_df = test_df.drop(columns=['Order_Placed_Date', 'Delivery_Date'])

# ============================================================
# Step 4: Split features and target (Your Logic)
# ============================================================
X = train_df.drop(columns=['Transport_Cost'])
y = train_df['Transport_Cost']
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# ============================================================
# Step 5: Preprocessing (Your Logic, better scaler)
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')), # Median is more robust
            ('scaler', RobustScaler()) # --- NEW: RobustScaler ---
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

# Define the model pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso()) # Placeholder
])

# ============================================================
# Step 6: Split data (Your Logic)
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# Step 7: Hyperparameter Optimization (NEW MODEL SEARCH)
# ============================================================
# We will search over all 3 models and their parameters at once
param_grid = [
    {
        'regressor': [Lasso(max_iter=10000)],
        'regressor__alpha': [0.1, 1, 10, 50, 100] # Alphas for Lasso
    },
    {
        'regressor': [Ridge(max_iter=10000)],
        'regressor__alpha': [1, 10, 50, 100, 200] # Alphas for Ridge
    },
    {
        'regressor': [ElasticNet(max_iter=10000)],
        'regressor__alpha': [0.1, 1, 10], # Alphas for ElasticNet
        'regressor__l1_ratio': [0.1, 0.5, 0.9] # L1/L2 mix
    }
]

print("Running GridSearchCV to find the best linear model (Lasso, Ridge, or ElasticNet)...")
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"\nBest Model and Parameters found:")
print(grid.best_params_)

# ============================================================
# Step 8: Evaluate on Validation Set
# ============================================================
best_model = grid.best_estimator_
val_preds = best_model.predict(X_val)
mae = mean_absolute_error(y_val, val_preds)
print(f"\nValidation MAE (on lucky split): {mae:.4f}")

# ============================================================
# Step 9: Predict on Test Data
# ============================================================
test_predictions = best_model.predict(test_df)

mean_positive_pred = test_predictions[test_predictions > 0].mean()
test_predictions = np.where(test_predictions < 0, mean_positive_pred, test_predictions)

# ============================================================
# Step 10: Create Submission File
# ============================================================
submission = pd.DataFrame({
    'Hospital_Id': test_ids_for_submission,
    'Transport_Cost': test_predictions
})
submission.to_csv('submission_robust_linear.csv', index=False)
print("\nSubmission file 'submission_robust_linear.csv' created successfully!")
print(submission.head())