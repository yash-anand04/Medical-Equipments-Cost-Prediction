import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Load the train and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Step 1: Replace negative values in Transport_Cost with 0
train_df.loc[train_df['Transport_Cost'] < 0, 'Transport_Cost'] = 0

# Step 2: Calculate mean of positive Transport_Cost values
mean_positive_cost = train_df.loc[train_df['Transport_Cost'] > 0, 'Transport_Cost'].mean()

# Step 3: Replace 0 values (original negative) with mean positive cost
train_df.loc[train_df['Transport_Cost'] == 0, 'Transport_Cost'] = mean_positive_cost

# Split features and target variable
X = train_df.drop(columns=['Transport_Cost'])
y = train_df['Transport_Cost']

# Identify numeric and categorical columns in features
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Define preprocessing pipeline
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

# Define LASSO regression pipeline
lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=0.1, random_state=42, max_iter=10000))
])

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train LASSO model
lasso_pipeline.fit(X_train, y_train)

# Validate model and calculate RMSE
val_preds = lasso_pipeline.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {rmse:.4f}")

# Predict costs for test set
test_predictions = lasso_pipeline.predict(test_df)

# Replace negative predictions with mean of positive predicted costs
mean_positive_test_pred = test_predictions[test_predictions > 0].mean()
test_predictions = np.where(test_predictions < 0, mean_positive_test_pred, test_predictions)

# Create submission DataFrame
submission = pd.DataFrame({
    'Customer_Id': test_df['Hospital_Id'],
    'Cost': test_predictions
})

# Save submission CSV
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
print(submission.head())
