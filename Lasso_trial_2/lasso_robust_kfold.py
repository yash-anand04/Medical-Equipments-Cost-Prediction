import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
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
y = train_df['Transport_Cost']

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

# ============================================================
# Step 6: K-Fold Cross-Validation (Replaces train_test_split)
# ============================================================
# We use the same best alpha your GridSearchCV found
BEST_ALPHA = 0.1 # This was the best in your script (0.0001 to 10)
                # You can change this if you found a different best alpha

N_SPLITS = 10
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_maes = []
test_predictions_list = []

print(f"Starting K-Fold training with your pipeline (Alpha={BEST_ALPHA})...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Define the *full* pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Lasso(alpha=BEST_ALPHA, max_iter=10000, random_state=42))
    ])
    
    # Fit the pipeline on this fold's training data
    pipeline.fit(X_train, y_train)
    
    # Evaluate on this fold's validation data
    val_preds = pipeline.predict(X_val)
    fold_mae = mean_absolute_error(y_val, val_preds)
    print(f"Fold MAE: {fold_mae:.2f}")
    oof_maes.append(fold_mae)
    
    # Predict on the *real* test set
    test_predictions_list.append(pipeline.predict(test_df))

print(f"\nOverall OOF MAE: {np.mean(oof_maes):.2f} +/- {np.std(oof_maes):.2f}")

# ============================================================
# Step 7: Average Test Predictions
# ============================================================
# Average the predictions from all 10 folds
final_test_predictions = np.mean(test_predictions_list, axis=0)

# Post-processing (Your logic)
mean_positive_pred = final_test_predictions[final_test_predictions > 0].mean()
final_test_predictions = np.where(final_test_predictions < 0, mean_positive_pred, final_test_predictions)

# ============================================================
# Step 8: Create Submission File
# ============================================================
submission = pd.DataFrame({
    'Hospital_Id': test_ids_for_submission,
    'Transport_Cost': final_test_predictions
})
submission.to_csv('submission_optimized_kfold.csv', index=False)
print("\nSubmission file 'submission_optimized_kfold.csv' created successfully!")
print(submission.head())