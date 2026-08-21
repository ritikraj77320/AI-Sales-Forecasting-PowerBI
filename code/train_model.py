import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

import joblib


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("data/ml_sales_data.csv")

print("=" * 60)
print("MACHINE LEARNING - SALES PREDICTION")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# =========================================================
# 2. SELECT FEATURES AND TARGET
# =========================================================

features = [
    "Quantity",
    "Unit_Price",
    "Discount",
    "Year",
    "Month",
    "Quarter",
    "Day",
    "Day_of_Week_Number",
    "Category",
    "Region"
]

target = "Sales"

X = df[features]
y = df[target]


# =========================================================
# 3. CATEGORICAL AND NUMERICAL FEATURES
# =========================================================

categorical_features = [
    "Category",
    "Region"
]

numerical_features = [
    "Quantity",
    "Unit_Price",
    "Discount",
    "Year",
    "Month",
    "Quarter",
    "Day",
    "Day_of_Week_Number"
]


# =========================================================
# 4. ENCODE CATEGORICAL FEATURES
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# =========================================================
# 5. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# =========================================================
# 6. CREATE MODELS
# =========================================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
}


# =========================================================
# 7. TRAIN AND EVALUATE MODELS
# =========================================================

results = {}

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"TRAINING: {name}")
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R²   : {r2:.4f}")


# =========================================================
# 8. DISPLAY MODEL COMPARISON
# =========================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results).T

print(results_df)


# =========================================================
# 9. SELECT BEST MODEL
# =========================================================

best_model_name = results_df["R2"].idxmax()

print("\nBest Model:")
print(best_model_name)


# =========================================================
# 10. TRAIN BEST MODEL AGAIN
# =========================================================

best_model = models[best_model_name]

best_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", best_model)]
)

best_pipeline.fit(X_train, y_train)


# =========================================================
# 11. SAVE BEST MODEL
# =========================================================

joblib.dump(best_pipeline, "models/sales_prediction_model.pkl")

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print("models/sales_prediction_model.pkl")