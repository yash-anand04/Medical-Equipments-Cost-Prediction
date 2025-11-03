import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LassoCV

from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')

# 1️⃣ Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Clip negative costs
negative_costs = train[train['Transport_Cost'] < 0]['Transport_Cost'].count()
if negative_costs > 0:
    print(f"Warning: Found {negative_costs} negative Transport_Cost values. Clipping them to 0.")
    train['Transport_Cost'] = train['Transport_Cost'].clip(lower=0)

# --- REMOVE EXTREME OUTLIERS ---
PERCENTILE_CLIP = 0.995 
clip_value = train['Transport_Cost'].quantile(PERCENTILE_CLIP)
print(f"Original training data shape: {train.shape}")
train = train[train['Transport_Cost'] < clip_value]
print(f"New training data shape after outlier clip: {train.shape}")
# --- END OUTLIER REMOVAL ---


# 2️⃣ Separate features and target
X = train.drop(columns=["Transport_Cost"]) 
test_for_processing = test.copy() 
test_ids_for_submission = test["Hospital_Id"]

y = train["Transport_Cost"]
y_log = np.log1p(y)

# 3️⃣ Advanced Preprocessing & Feature Engineering
print("Starting advanced feature engineering...")
combined = pd.concat([X, test_for_processing], axis=0)

# --- DATE FEATURES ---
combined["Order_Placed_Date"] = pd.to_datetime(combined["Order_Placed_Date"], errors="coerce")
combined["Delivery_Date"] = pd.to_datetime(combined["Delivery_Date"], errors="coerce")
combined["Delivery_Duration_Days"] = (combined["Delivery_Date"] - combined["Order_Placed_Date"]).dt.days

# --- Cyclical Features ---
combined['Order_Month'] = combined['Order_Placed_Date'].dt.month
combined['Order_DayOfWeek'] = combined['Order_Placed_Date'].dt.dayofweek
combined['Order_Month_sin'] = np.sin(2 * np.pi * combined['Order_Month']/12)
combined['Order_Month_cos'] = np.cos(2 * np.pi * combined['Order_Month']/12)
combined['Order_DayOfWeek_sin'] = np.sin(2 * np.pi * combined['Order_DayOfWeek']/7)
combined['Order_DayOfWeek_cos'] = np.cos(2 * np.pi * combined['Order_DayOfWeek']/7)
combined["Order_Is_Weekend"] = (combined['Order_DayOfWeek'] >= 5).astype(int)
combined.drop(columns=["Order_Placed_Date", "Delivery_Date"], inplace=True)

# --- IMPUTATION (Numerical) ---
duration_median = combined['Delivery_Duration_Days'].median()
combined['Delivery_Duration_Days'] = combined['Delivery_Duration_Days'].fillna(duration_median)

# --- IMPUTATION (Categorical) ---
cat_cols_raw = combined.select_dtypes(include=["object", "category"]).columns
for col in cat_cols_raw:
    combined[col] = combined[col].fillna("Missing")

# --- AGGREGATION & INTERACTION FEATURES ---
hospital_agg_feats = combined.groupby('Hospital_Id')['Delivery_Duration_Days'].agg(['mean', 'std']).reset_index()
hospital_agg_feats.columns = ['Hospital_Id', 'Hospital_Dur_Mean', 'Hospital_Dur_Std']
hospital_order_count = combined['Hospital_Id'].value_counts().reset_index()
hospital_order_count.columns = ['Hospital_Id', 'Hospital_Order_Count']
combined = combined.merge(hospital_agg_feats, on='Hospital_Id', how='left')
combined = combined.merge(hospital_order_count, on='Hospital_Id', how='left')
combined['Hospital_Dur_Std'] = combined['Hospital_Dur_Std'].fillna(0)
combined['Duration_vs_Hospital_Mean'] = combined['Delivery_Duration_Days'] / combined['Hospital_Dur_Mean']
combined.replace([np.inf, -np.inf], np.nan, inplace=True)
num_cols = combined.select_dtypes(include=np.number).columns
for col in num_cols:
    if combined[col].isnull().any():
        median_val = combined[col].median()
        combined[col] = combined[col].fillna(median_val)
# --- END FE ---


# --- 4️⃣ Prepare Data for THREE Models ---
cat_features_list = ['Hospital_Id', 'Supplier_Name', 'Equipment_Type', 'CrossBorder_Shipping', 
                     'Urgent_Shipping', 'Installation_Service', 'Transport_Method', 
                     'Fragile_Equipment', 'Hospital_Info', 'Rural_Hospital', 'Hospital_Location', 
                     'Order_Month', 'Order_DayOfWeek', 'Order_Is_Weekend']
cat_features_list = [col for col in cat_features_list if col in combined.columns]
print(f"Found {len(cat_features_list)} categorical features: {cat_features_list}")

# --- Prep for CatBoost (Strings) ---
X_processed_catboost = combined.iloc[:len(X), :].copy() 
X_test_processed_catboost = combined.iloc[len(X):, :].copy()
for col in cat_features_list:
    X_processed_catboost[col] = X_processed_catboost[col].astype(str)
    X_test_processed_catboost[col] = X_test_processed_catboost[col].astype(str)

# --- Prep for LGBM & Lasso (Label Encoded) ---
print("Label Encoding for LightGBM & Lasso...")
combined_encoded = combined.copy()
le_hospital = LabelEncoder()
combined_encoded['Hospital_Id_encoded'] = le_hospital.fit_transform(combined_encoded['Hospital_Id'].astype(str))
lgbm_cat_features = []
for col in cat_features_list:
    if col in combined_encoded.columns:
        combined_encoded[col] = combined_encoded[col].astype(str)
        le = LabelEncoder()
        combined_encoded[col] = le.fit_transform(combined_encoded[col])
        lgbm_cat_features.append(col) 
        
X_processed_encoded = combined_encoded.iloc[:len(X), :]
X_test_processed_encoded = combined_encoded.iloc[len(X):, :]

numeric_cols = [col for col in X_processed_encoded.columns if col not in lgbm_cat_features and col != 'Hospital_Id_encoded']
# --- End Data Prep ---


# 5️⃣ K-Fold Cross-Validation with ENSEMBLE
print("Starting K-Fold Ensemble Training...")
N_SPLITS = 10 
kf = GroupKFold(n_splits=N_SPLITS)
groups = X_processed_encoded['Hospital_Id_encoded'] 

test_preds_lgbm_list = []
test_preds_catboost_list = []
test_preds_lasso_list = [] 
oof_maes_lgbm = []
oof_maes_catboost = []
oof_maes_lasso = [] 

# --- ⭐️ YOUR TUNED PARAMS ARE NOW PLUGGED IN ⭐️ ---
LGBM_BEST_PARAMS = {
    'objective': 'regression_l1',
    'n_estimators': 2000,
    'learning_rate': 0.09819679607511958, # Tuned
    'num_leaves': 31,                     # Tuned
    'reg_alpha': 0.013515022360853321,    # Tuned
    'reg_lambda': 0.19525120348000968,   # Tuned
    'subsample': 0.7477466474043304,      # Tuned
    'colsample_bytree': 0.6015789021330963,# Tuned
    'random_state': 42,
    'n_jobs': -1
}

CATBOOST_BEST_PARAMS = {
    'iterations': 2000,
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'learning_rate': 0.08193442210707869, # Tuned
    'l2_leaf_reg': 9.816086112240033,   # Tuned
    'depth': 4,                           # Tuned
    'random_seed': 42,
    'verbose': 0
}
# --- ⭐️ END PARAMS ⭐️ ---


for fold, (train_index, val_index) in enumerate(kf.split(X_processed_encoded, y_log, groups=groups)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    y_train_fold, y_val_fold = y_log.iloc[train_index], y_log.iloc[val_index]
    y_val_fold_orig = y.iloc[val_index]

    # --- 1. LightGBM (Uses Tuned Params) ---
    print("Training LGBM...")
    X_train_lgbm, X_val_lgbm = X_processed_encoded.iloc[train_index], X_processed_encoded.iloc[val_index]
    model_lgbm = LGBMRegressor(**LGBM_BEST_PARAMS) # Use the tuned params
    model_lgbm.fit(X_train_lgbm, y_train_fold, eval_set=[(X_val_lgbm, y_val_fold)],
                   eval_metric='mae', callbacks=[early_stopping(100, verbose=False)], # Shortened patience
                   categorical_feature=lgbm_cat_features)
    val_preds_lgbm = np.expm1(model_lgbm.predict(X_val_lgbm))
    fold_mae_lgbm = mean_absolute_error(y_val_fold_orig, val_preds_lgbm)
    print(f"LGBM Fold {fold+1} MAE: {fold_mae_lgbm:.2f}")
    oof_maes_lgbm.append(fold_mae_lgbm)
    test_preds_lgbm_list.append(np.expm1(model_lgbm.predict(X_test_processed_encoded)))

    # --- 2. CatBoost (Uses Tuned Params) ---
    print("Training CatBoost...")
    X_train_cat, X_val_cat = X_processed_catboost.iloc[train_index], X_processed_catboost.iloc[val_index]
    model_catboost = CatBoostRegressor(**CATBOOST_BEST_PARAMS) # Use the tuned params
    model_catboost.fit(X_train_cat, y_train_fold, eval_set=[(X_val_cat, y_val_fold)],
                       cat_features=cat_features_list, early_stopping_rounds=100) # Shortened patience
    val_preds_catboost = np.expm1(model_catboost.predict(X_val_cat))
    fold_mae_catboost = mean_absolute_error(y_val_fold_orig, val_preds_catboost)
    print(f"CatBoost Fold {fold+1} MAE: {fold_mae_catboost:.2f}")
    oof_maes_catboost.append(fold_mae_catboost)
    test_preds_catboost_list.append(np.expm1(model_catboost.predict(X_test_processed_catboost)))

    # --- 3. LassoCV (Tunes Itself) ---
    print("Training LassoCV...")
    X_train_enc, X_val_enc = X_processed_encoded.iloc[train_index], X_processed_encoded.iloc[val_index]
    X_test_enc = X_test_processed_encoded.copy()
    
    scaler = StandardScaler()
    X_train_enc[numeric_cols] = scaler.fit_transform(X_train_enc[numeric_cols])
    X_val_enc[numeric_cols] = scaler.transform(X_val_enc[numeric_cols])
    X_test_enc[numeric_cols] = scaler.transform(X_test_enc[numeric_cols])
    
    model_lasso = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=2000)
    model_lasso.fit(X_train_enc, y_train_fold)
    
    val_preds_lasso = np.expm1(model_lasso.predict(X_val_enc))
    val_preds_lasso[val_preds_lasso < 0] = 0 
    fold_mae_lasso = mean_absolute_error(y_val_fold_orig, val_preds_lasso)
    print(f"LassoCV Fold {fold+1} MAE: {fold_mae_lasso:.2f}")
    oof_maes_lasso.append(fold_mae_lasso)
    
    test_preds_fold_lasso = np.expm1(model_lasso.predict(X_test_enc))
    test_preds_fold_lasso[test_preds_fold_lasso < 0] = 0
    test_preds_lasso_list.append(test_preds_fold_lasso)
    # --- END MODEL ---

print("---" * 15)
print(f"Overall OOF LGBM MAE: {np.mean(oof_maes_lgbm):.2f} +/- {np.std(oof_maes_lgbm):.2f}")
print(f"Overall OOF CatBoost MAE: {np.mean(oof_maes_catboost):.2f} +/- {np.std(oof_maes_catboost):.2f}")
print(f"Overall OOF LassoCV MAE: {np.mean(oof_maes_lasso):.2f} +/- {np.std(oof_maes_lasso):.2f}")

# 6️⃣ Average test predictions (ENSEMBLE)
print("Averaging ensemble predictions...")
final_test_preds_lgbm = np.mean(test_preds_lgbm_list, axis=0)
final_test_preds_catboost = np.mean(test_preds_catboost_list, axis=0)
final_test_preds_lasso = np.mean(test_preds_lasso_list, axis=0)

# Using the weighted ensemble that favors Lasso
final_test_predictions = (final_test_preds_lgbm * 0.25) + \
                         (final_test_preds_catboost * 0.25) + \
                         (final_test_preds_lasso * 0.50)
print("Using weighted ensemble: 25% LGBM, 25% CatBoost, 50% LassoCV")


# 7️⃣ Create submission file
submission = pd.DataFrame({
    "Hospital_Id": test_ids_for_submission,
    "Transport_Cost": final_test_predictions
})

# ✅ Save safely
output_path = "submission_tuned_ensemble_v2.csv" # New name
submission.to_csv(output_path, index=False)
print(f"✅ {output_path} saved successfully!")
print(submission.shape)
print(submission.head())