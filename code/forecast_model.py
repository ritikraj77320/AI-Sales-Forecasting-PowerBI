import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


# =========================================================
# 1. LOAD FORECASTING DATA
# =========================================================

df = pd.read_csv("data/forecasting_sales_data.csv")

df["Month_Date"] = pd.to_datetime(df["Month_Date"])

print("=" * 70)
print("TIME-BASED SALES FORECASTING")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)


# =========================================================
# 2. SORT BY DATE
# =========================================================

df = df.sort_values("Month_Date").reset_index(drop=True)


# =========================================================
# 3. DEFINE FEATURES
# =========================================================

features = [
    "Sales_Lag_1",
    "Sales_Lag_2",
    "Sales_Lag_3",
    "Rolling_3_Month_Average",
    "Sales_Growth",
    "Year",
    "Month",
    "Quarter",
    "Category",
    "Region"
]

target = "Sales"


X = df[features]
y = df[target]


# =========================================================
# 4. TIME-BASED TRAIN / TEST SPLIT
# =========================================================

# Use the first 80% of time for training
# and the final 20% for testing.

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))

print("\nTraining period:")
print(
    df["Month_Date"].iloc[:split_index].min(),
    "to",
    df["Month_Date"].iloc[:split_index].max()
)

print("\nTesting period:")
print(
    df["Month_Date"].iloc[split_index:].min(),
    "to",
    df["Month_Date"].iloc[split_index:].max()
)


# =========================================================
# 5. CATEGORICAL FEATURES
# =========================================================

categorical_features = [
    "Category",
    "Region"
]


# =========================================================
# 6. PREPROCESSING
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
# 7. CREATE MODELS
# =========================================================

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
}


# =========================================================
# 8. TRAIN MODELS
# =========================================================

results = {}

trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 70)
    print("TRAINING:", name)
    print("=" * 70)

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
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    trained_models[name] = pipeline

    print(f"MAE  : ₹{mae:,.2f}")
    print(f"RMSE : ₹{rmse:,.2f}")
    print(f"R²   : {r2:.4f}")


# =========================================================
# 9. MODEL COMPARISON
# =========================================================

print("\n" + "=" * 70)
print("FORECASTING MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results).T

print(results_df)


# =========================================================
# 10. FIND BEST MODEL
# =========================================================

best_model_name = results_df["R2"].idxmax()

print("\nBest Forecasting Model:")
print(best_model_name)


# =========================================================
# 11. SAVE BEST MODEL
# =========================================================

best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    "models/sales_forecasting_model.pkl"
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    "models/sales_forecasting_model.pkl"
)


# =========================================================
# 12. CREATE PREDICTION DATAFRAME
# =========================================================

predictions = best_model.predict(X_test)

prediction_df = pd.DataFrame({
    "Month_Date": df["Month_Date"].iloc[split_index:].values,
    "Category": df["Category"].iloc[split_index:].values,
    "Region": df["Region"].iloc[split_index:].values,
    "Actual_Sales": y_test.values,
    "Predicted_Sales": predictions
})


# =========================================================
# 13. CALCULATE ERROR
# =========================================================

prediction_df["Error"] = (
    prediction_df["Actual_Sales"]
    - prediction_df["Predicted_Sales"]
)

prediction_df["Absolute_Error"] = (
    prediction_df["Error"].abs()
)


# =========================================================
# 14. SAVE PREDICTIONS
# =========================================================

prediction_df.to_csv(
    "data/forecast_predictions.csv",
    index=False
)

print("\nPrediction file saved:")
print("data/forecast_predictions.csv")


# =========================================================
# 15. DISPLAY SAMPLE PREDICTIONS
# =========================================================

print("\n" + "=" * 70)
print("SAMPLE FORECAST PREDICTIONS")
print("=" * 70)

print(
    prediction_df.head(15).to_string(index=False)
)