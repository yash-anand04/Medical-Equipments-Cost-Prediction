# improved_pipeline_for_kaggle.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# optional libs
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

try:
    from category_encoders import TargetEncoder
    CAT_ENC_AVAILABLE = True
except Exception:
    CAT_ENC_AVAILABLE = False

RANDOM_STATE = 42
N_SPLITS = 5

# -------------------------
# Load
# -------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_ids_for_submission = test['Hospital_Id'].copy()

# -------------------------
# Fix/clean Transport_Cost (training-target-level)
# - Replace negative with NaN so we can handle via median/transform below
# -------------------------
train.loc[train['Transport_Cost'] < 0, 'Transport_Cost'] = np.nan

# We'll use log1p target transform to stabilize variance
# Fill zeros/NaNs temporarily with median positive for log transform (we'll do better OOF below)
median_positive = train['Transport_Cost'].loc[train['Transport_Cost'] > 0].median()
train['Transport_Cost'] = train['Transport_Cost'].fillna(median_positive)

# Apply log1p transform
train['y'] = np.log1p(train['Transport_Cost'])

# -------------------------
# Date processing: create features BEFORE dropping
# Swap if order>delivery like your original logic
# -------------------------
def process_dates_create_feats(df):
    df = df.copy()
    df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], errors='coerce')
    df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')

    # swap when order > delivery
    invalid_mask = df['Order_Placed_Date'] > df['Delivery_Date']
    df.loc[invalid_mask, ['Order_Placed_Date', 'Delivery_Date']] = \
        df.loc[invalid_mask, ['Delivery_Date', 'Order_Placed_Date']].values

    # durations and components
    df['Delivery_Duration'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days
    # if negative / zero -> set to median later
    df['Order_day'] = df['Order_Placed_Date'].dt.day
    df['Order_month'] = df['Order_Placed_Date'].dt.month
    df['Order_weekday'] = df['Order_Placed_Date'].dt.weekday
    df['Delivery_day'] = df['Delivery_Date'].dt.day
    df['Delivery_month'] = df['Delivery_Date'].dt.month
    df['Delivery_weekday'] = df['Delivery_Date'].dt.weekday

    return df

train = process_dates_create_feats(train)
test = process_dates_create_feats(test)

# fill Delivery_Duration <=0 with median positive duration
median_duration = train.loc[train['Delivery_Duration'] > 0, 'Delivery_Duration'].median()
train.loc[train['Delivery_Duration'] <= 0, 'Delivery_Duration'] = median_duration
train['Delivery_Duration'].fillna(median_duration, inplace=True)
test.loc[test['Delivery_Duration'] <= 0, 'Delivery_Duration'] = median_duration
test['Delivery_Duration'].fillna(median_duration, inplace=True)

# Drop raw date columns (we already derived features)
train = train.drop(columns=['Order_Placed_Date', 'Delivery_Date', 'Transport_Cost'])
test = test.drop(columns=['Order_Placed_Date', 'Delivery_Date'])

# -------------------------
# Basic feature engineering: counts & frequency encodings
# -------------------------
def add_count_features(df, cols):
    for c in cols:
        cnt = df[c].map(df[c].value_counts())
        df[f"{c}_count"] = cnt
    return df

# choose categorical columns (strings) and numeric
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
num_cols = train.select_dtypes(include=['int64', 'float64']).drop(columns=['y']).columns.tolist()

# Add count features for categorical cols (both train & test)
combined = pd.concat([train.drop(columns=['y']), test], axis=0, ignore_index=True)
for c in cat_cols:
    combined[f"{c}_count"] = combined[c].map(combined[c].value_counts())

# split combined back
train_feat = combined.iloc[:len(train)].copy()
test_feat = combined.iloc[len(train):].copy()
train = pd.concat([train_feat.reset_index(drop=True), train.reset_index(drop=True)['y']], axis=1)
test = test_feat.reset_index(drop=True)

# Also add simple numeric interactions (if many numeric)
if len(num_cols) >= 2:
    # pairwise ratios for top 3 numeric columns (defensive)
    top_nums = num_cols[:3]
    for i in range(len(top_nums)):
        for j in range(i+1, len(top_nums)):
            a, b = top_nums[i], top_nums[j]
            train[f"{a}_div_{b}"] = train[a] / (train[b].replace(0, np.nan))
            test[f"{a}_div_{b}"] = test[a] / (test[b].replace(0, np.nan))

# Recompute lists
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
num_cols = train.select_dtypes(include=['int64', 'float64']).drop(columns=['y']).columns.tolist()

# -------------------------
# Target-encoding for categorical features (out-of-fold)
# If category_encoders.TargetEncoder available, we will use custom OOF target encoding to avoid leakage
# -------------------------
def oof_target_encode(train_df, test_df, cols, target, n_splits=5, seed=RANDOM_STATE):
    """Return (train_encoded_df, test_encoded_df) with new cols {col}_te"""
    train_out = train_df.copy()
    test_out = test_df.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for col in cols:
        oof = np.zeros(len(train_df))
        test_col_vals = np.zeros((len(test_df), n_splits))
        for i, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
            tr, val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
            means = tr.groupby(col)[target].mean()
            # map
            oof[val_idx] = train_df[col].iloc[val_idx].map(means).fillna(means.mean())
            test_col_vals[:, i] = test_df[col].map(means).fillna(means.mean())
        train_out[f"{col}_te"] = oof
        test_out[f"{col}_te"] = test_col_vals.mean(axis=1)
    return train_out, test_out

# apply OOF target-encoding to categorical columns that have moderate cardinality
# choose columns with unique values less than, say, 1000 (to be safe)
te_cols = [c for c in cat_cols if train[c].nunique() < 1000]
if len(te_cols) > 0:
    train, test = oof_target_encode(train, test, te_cols, target='y', n_splits=N_SPLITS)

# After encoding, we can drop raw cat columns or keep both
# Keep both for now but the modeling pipeline will choose numeric transforms
# Replace remaining object columns in numeric pipeline by imputation/ordinal later (we will one-hot small-cardinality)
# -------------------------

# -------------------------
# Preprocessing pipelines for numerical and categorical
# -------------------------
# We'll use RobustScaler for numeric to reduce outlier influence
numeric_features = [c for c in train.select_dtypes(include=[np.number]).columns.tolist() if c != 'y']
# For categorical, keep remaining object columns with small cardinality for one-hot
cat_small = [c for c in cat_cols if train[c].nunique() <= 20]  # one-hot these
cat_large = [c for c in cat_cols if train[c].nunique() > 20]   # leave as-is (we already did TE)

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

# small-cardinality one-hot
cat_small_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# large cardinality ordinal (fallback) - if we didn't TE them, we encode ordinally
cat_large_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat_small', cat_small_transformer, cat_small),
        ('cat_large', cat_large_transformer, cat_large)
    ],
    remainder='drop'  # ignore other cols
)

# -------------------------
# Models and stacking
# -------------------------
# Base estimators: LightGBM (if available) and Lasso
def make_lgb_regressor(random_state=RANDOM_STATE):
    if LGB_AVAILABLE:
        return lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        # fallback to sklearn's gradient boosting
        return HistGradientBoostingRegressor(random_state=random_state, max_iter=800)

lgb_model = make_lgb_regressor()
lasso_model = Lasso(alpha=0.01, max_iter=10000, random_state=RANDOM_STATE)

# Full pipeline for each base model (preprocessor + model)
from sklearn.pipeline import make_pipeline
pipe_lgb = make_pipeline(preprocessor, lgb_model)
pipe_lasso = make_pipeline(preprocessor, lasso_model)

# Stacking regressor (use simple linear blender)
estimators = [('lgb', pipe_lgb), ('lasso', pipe_lasso)]
stack = StackingRegressor(estimators=estimators, final_estimator=Lasso(alpha=0.01, max_iter=10000), n_jobs=-1, passthrough=False)

# -------------------------
# Out-of-fold stacking predictions to create ensemble
# -------------------------
def get_oof_preds(model, X, y, X_test, n_splits=5, random_state=RANDOM_STATE):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_train = np.zeros(len(X))
    oof_test = np.zeros(len(X_test))
    test_preds_folds = np.zeros((len(X_test), n_splits))
    for i, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        oof_train[val_idx] = model.predict(X_val)
        test_preds_folds[:, i] = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_val, oof_train[val_idx]))
        print(f"Fold {i+1} RMSE: {rmse:.6f}")
    oof_test = test_preds_folds.mean(axis=1)
    return oof_train, oof_test

# Prepare X and y (y is log1p target)
X = train.drop(columns=['y']).reset_index(drop=True)
y = train['y'].reset_index(drop=True)
X_test = test.reset_index(drop=True)

# First: get OOF preds for LGB and Lasso separately
print("Generating OOF predictions for LightGBM (or fallback)...")
oof_lgb_train, oof_lgb_test = get_oof_preds(pipe_lgb, X, y, X_test, n_splits=N_SPLITS)

print("Generating OOF predictions for Lasso...")
oof_lasso_train, oof_lasso_test = get_oof_preds(pipe_lasso, X, y, X_test, n_splits=N_SPLITS)

# Simple ensemble: weighted average by inverse validation RMSE (approx)
lgb_rmse = np.sqrt(mean_squared_error(y, oof_lgb_train))
lasso_rmse = np.sqrt(mean_squared_error(y, oof_lasso_train))
w_lgb = 1 / lgb_rmse
w_lasso = 1 / lasso_rmse
w_sum = w_lgb + w_lasso
w_lgb /= w_sum
w_lasso /= w_sum
print(f"Weights (LGB, Lasso): {w_lgb:.3f}, {w_lasso:.3f}")

oof_ensemble_train = w_lgb * oof_lgb_train + w_lasso * oof_lasso_train
oof_ensemble_test = w_lgb * oof_lgb_test + w_lasso * oof_lasso_test
ensemble_rmse = np.sqrt(mean_squared_error(y, oof_ensemble_train))
print(f"Ensemble OOF RMSE (log-scale): {ensemble_rmse:.6f}")

# Optionally: fit stacking final blender on OOF predictions (meta-model) - quick 2nd-level stack
meta_X = pd.DataFrame({
    'lgb_oof': oof_lgb_train,
    'lasso_oof': oof_lasso_train
})
meta_test = pd.DataFrame({
    'lgb_oof': oof_lgb_test,
    'lasso_oof': oof_lasso_test
})

meta_model = Lasso(alpha=0.01, max_iter=10000, random_state=RANDOM_STATE)
meta_model.fit(meta_X, y)
meta_oof = meta_model.predict(meta_X)
meta_test_pred = meta_model.predict(meta_test)
meta_rmse = np.sqrt(mean_squared_error(y, meta_oof))
print(f"Meta-model RMSE (log-scale): {meta_rmse:.6f}")

# Choose final predictions (I will average ensemble and meta-model preds in log space)
final_test_log = 0.5 * oof_ensemble_test + 0.5 * meta_test_pred

# Convert back from log1p
final_test = np.expm1(final_test_log)

# ensure no negative predictions; replace negatives by median of positive predictions
mean_positive_pred = final_test[final_test > 0].mean()
final_test = np.where(final_test < 0, mean_positive_pred, final_test)

# create submission
submission = pd.DataFrame({
    "Hospital_Id": test_ids_for_submission,
    "Transport_Cost": final_test
})

submission.to_csv("submission_improved_ensemble.csv", index=False)
print("Saved submission_improved_ensemble.csv")
print(submission.head())
