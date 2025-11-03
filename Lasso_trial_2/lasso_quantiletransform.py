import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ============================================================
# Step 1: Load datasets
# ============================================================
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

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
# Step 4: Split features and target
# ============================================================
X = train_df.drop(columns=['Transport_Cost'])
y = train_df['Transport_Cost'] # This is the original, untransformed y

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

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

# Define the LASSO model within pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso(max_iter=10000))
])

# ============================================================
# Step 6: Split data (Your Logic)
# ============================================================
X_train, X_val, y_train_orig, y_val_orig = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# Step 7: !!! NEW - "TAME" THE TARGET VARIABLE !!!
# ============================================================
print("Applying QuantileTransformer to 'tame' the target variable...")
# n_quantiles=1000 is high-resolution; random_state=42 for consistency
qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)

# Fit *only* on the training y data
# We must reshape y to be a 2D array
y_train_transformed = qt.fit_transform(y_train_orig.values.reshape(-1, 1)).ravel()

# Transform the validation y data
y_val_transformed = qt.transform(y_val_orig.values.reshape(-1, 1)).ravel()

# ============================================================
# Step 8: Hyperparameter Optimization
# ============================================================
param_grid = {'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit on X_train and the *transformed* y_train
print("Running GridSearchCV on the 'tamed' target...")
grid.fit(X_train, y_train_transformed)

best_alpha = grid.best_params_['regressor__alpha']
print(f"Best Alpha (from GridSearchCV): {best_alpha}")

# ============================================================
# Step 9: Evaluate on Validation Set
# ============================================================
best_model = grid.best_estimator_
# Predict the *transformed* values
val_preds_transformed = best_model.predict(X_val)

# !!! IMPORTANT: Inverse transform predictions back to dollars !!!
val_preds_orig = qt.inverse_transform(val_preds_transformed.reshape(-1, 1)).ravel()

# Now we can calculate the real RMSE and MAE
rmse = np.sqrt(mean_squared_error(y_val_orig, val_preds_orig))
mae = mean_absolute_error(y_val_orig, val_preds_orig) # MAE is more robust
print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation MAE: {mae:.4f}") # This should be much more stable

# ============================================================
# Step 10: Predict on Test Data
# ============================================================
print("Predicting on test data...")
# Predict the *transformed* values
test_preds_transformed = best_model.predict(test_df)

# !!! IMPORTANT: Inverse transform predictions back to dollars !!!
test_predictions = qt.inverse_transform(test_preds_transformed.reshape(-1, 1)).ravel()

# Post-processing (Your logic)
mean_positive_pred = test_predictions[test_predictions > 0].mean()
test_predictions = np.where(test_predictions < 0, mean_positive_pred, test_predictions)

# ============================================================
# Step 11: Create Submission File
# ============================================================
submission = pd.DataFrame({
    'Hospital_Id': test_ids_for_submission,
    'Transport_Cost': test_predictions
})
submission.to_csv('submission_quantile_lasso.csv', index=False)
print("\nSubmission file 'submission_quantile_lasso.csv' created successfully!")
print(submission.head())