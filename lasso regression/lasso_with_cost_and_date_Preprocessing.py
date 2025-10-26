import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ============================================================
# Step 1: Load datasets
# ============================================================
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# ============================================================
# Step 2: Fix negative Transport_Cost
# ============================================================
train_df.loc[train_df['Transport_Cost'] < 0, 'Transport_Cost'] = 0
mean_positive_cost = train_df.loc[train_df['Transport_Cost'] > 0, 'Transport_Cost'].mean()
train_df.loc[train_df['Transport_Cost'] == 0, 'Transport_Cost'] = mean_positive_cost

# ============================================================
# Step 3: Process dates with swapping invalid pairs
# ============================================================
def process_dates_with_swap(df):
    df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], errors='coerce')
    df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')

    # Identify invalid rows where order date is after delivery date
    invalid_dates_mask = df['Order_Placed_Date'] > df['Delivery_Date']

    # Swap those dates
    df.loc[invalid_dates_mask, ['Order_Placed_Date', 'Delivery_Date']] = \
        df.loc[invalid_dates_mask, ['Delivery_Date', 'Order_Placed_Date']].values

    # Calculate delivery duration in days
    df['Delivery_Duration'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days

    # Replace invalid or missing durations with mean positive duration
    mean_duration = df.loc[df['Delivery_Duration'] > 0, 'Delivery_Duration'].mean()
    df.loc[df['Delivery_Duration'] <= 0, 'Delivery_Duration'] = mean_duration
    df['Delivery_Duration'].fillna(mean_duration, inplace=True)

    return df

# Apply date processing
train_df = process_dates_with_swap(train_df)
test_df = process_dates_with_swap(test_df)

# Drop original date columns if not needed further
train_df = train_df.drop(columns=['Order_Placed_Date', 'Delivery_Date'])
test_df = test_df.drop(columns=['Order_Placed_Date', 'Delivery_Date'])

# ============================================================
# Step 4: Separate features and target
# ============================================================
X = train_df.drop(columns=['Transport_Cost'])
y = train_df['Transport_Cost']

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# ============================================================
# Step 5: Setup preprocessing pipeline
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

# ============================================================
# Step 6: LASSO regression pipeline
# ============================================================
lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=0.1, random_state=42, max_iter=10000))
])

# ============================================================
# Step 7: Train-validation split and model training
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lasso_pipeline.fit(X_train, y_train)

# Evaluate on validation set
val_preds = lasso_pipeline.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {rmse:.4f}")

# ============================================================
# Step 8: Predict on test data
# ============================================================
test_predictions = lasso_pipeline.predict(test_df)

# Replace any negative predictions with mean of positive predictions
mean_positive_pred = test_predictions[test_predictions > 0].mean()
test_predictions = np.where(test_predictions < 0, mean_positive_pred, test_predictions)

# ============================================================
# Step 9: Prepare submission file
# ============================================================
submission = pd.DataFrame({
    'Customer_Id': test_df['Hospital_Id'],
    'Cost': test_predictions
})

submission.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print(submission.head())
