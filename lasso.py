import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Feature-target separation (no changes to Transport_Cost)
X = train_df.drop(columns=['Transport_Cost'])
y = train_df['Transport_Cost']

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipeline
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

# LASSO regression pipeline
lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=0.1, random_state=42, max_iter=10000))
])

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model using unmodified train target values
lasso_pipeline.fit(X_train, y_train)

# Validate model and compute RMSE
val_preds = lasso_pipeline.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {rmse:.4f}")

# Predict test set
test_predictions = lasso_pipeline.predict(test_df)

# Note: No replacement of negative predictions performed here

# Create submission DataFrame
submission = pd.DataFrame({
    'Customer_Id': test_df['Hospital_Id'],
    'Cost': test_predictions
})

# Save submission CSV
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
print(submission.head())
