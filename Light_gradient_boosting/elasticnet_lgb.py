import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# -------------------------------
# 1. Load data
# -------------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_ids = test["Hospital_Id"]

# -------------------------------
# 2. Clean Transport_Cost
# -------------------------------
train.loc[train["Transport_Cost"] < 0, "Transport_Cost"] = np.nan
mean_positive = train.loc[train["Transport_Cost"] > 0, "Transport_Cost"].mean()
train["Transport_Cost"].fillna(mean_positive, inplace=True)

# -------------------------------
# 3. Process dates (same logic as before)
# -------------------------------
def process_dates(df):
    df = df.copy()
    df["Order_Placed_Date"] = pd.to_datetime(df["Order_Placed_Date"], errors="coerce")
    df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"], errors="coerce")

    invalid = df["Order_Placed_Date"] > df["Delivery_Date"]
    df.loc[invalid, ["Order_Placed_Date", "Delivery_Date"]] = \
        df.loc[invalid, ["Delivery_Date", "Order_Placed_Date"]].values

    df["Delivery_Duration"] = (df["Delivery_Date"] - df["Order_Placed_Date"]).dt.days
    df["Order_Month"] = df["Order_Placed_Date"].dt.month
    df["Order_Weekday"] = df["Order_Placed_Date"].dt.weekday
    df["Delivery_Month"] = df["Delivery_Date"].dt.month
    df["Delivery_Weekday"] = df["Delivery_Date"].dt.weekday

    median_dur = df.loc[df["Delivery_Duration"] > 0, "Delivery_Duration"].median()
    df["Delivery_Duration"].fillna(median_dur, inplace=True)
    df.loc[df["Delivery_Duration"] <= 0, "Delivery_Duration"] = median_dur

    return df.drop(columns=["Order_Placed_Date", "Delivery_Date"])

train = process_dates(train)
test = process_dates(test)

# -------------------------------
# 4. Split X, y
# -------------------------------
X = train.drop(columns=["Transport_Cost"])
y = train["Transport_Cost"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# -------------------------------
# 5. Preprocessor
# -------------------------------
from sklearn import __version__ as skl_version
if skl_version >= "1.2":
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", ohe)
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])

# -------------------------------
# 6. Model
# -------------------------------
model = ElasticNet(max_iter=10000, random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

param_grid = {
    "model__alpha": np.logspace(-3, 1, 30),
    "model__l1_ratio": np.linspace(0.1, 0.9, 9)
}

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)

best_model = grid.best_estimator_

val_preds = best_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {rmse:.4f}")

# -------------------------------
# 7. Predict Test
# -------------------------------
test_preds = best_model.predict(test)
test_preds = np.where(test_preds < 0, mean_positive, test_preds)

submission = pd.DataFrame({
    "Hospital_Id": test_ids,
    "Transport_Cost": test_preds
})
submission.to_csv("submission_elasticnet_poly.csv", index=False)
print("✅ submission_elasticnet_poly.csv created")
