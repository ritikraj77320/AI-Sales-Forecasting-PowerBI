import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/ml_sales_data.csv")

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


# ==========================================
# 2. CATEGORICAL FEATURES
# ==========================================

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


# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 4. PREPROCESSING
# ==========================================

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


# ==========================================
# 5. RANDOM FOREST
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# 6. TRAIN
# ==========================================

pipeline.fit(X_train, y_train)


# ==========================================
# 7. PREDICTION
# ==========================================

predictions = pipeline.predict(X_test)


# ==========================================
# 8. EVALUATION
# ==========================================

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

print("=" * 60)
print("RANDOM FOREST EVALUATION")
print("=" * 60)

print(f"MAE  : ₹{mae:,.2f}")
print(f"RMSE : ₹{rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# ==========================================
# 9. ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.5
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

# Perfect prediction line
min_value = min(y_test.min(), predictions.min())
max_value = max(y_test.max(), predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value]
)

plt.tight_layout()
plt.show()


# ==========================================
# 10. RESIDUALS
# ==========================================

residuals = y_test - predictions

plt.figure(figsize=(10, 6))

plt.scatter(predictions, residuals, alpha=0.5)

plt.axhline(y=0)

plt.xlabel("Predicted Sales")
plt.ylabel("Residual")
plt.title("Prediction Errors")

plt.tight_layout()
plt.show()