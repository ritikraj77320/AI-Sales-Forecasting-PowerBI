import pandas as pd
import numpy as np
import joblib

# =========================================================
# 1. LOAD DATA AND MODEL
# =========================================================

df = pd.read_csv("data/forecasting_sales_data.csv")

df["Month_Date"] = pd.to_datetime(df["Month_Date"])

model = joblib.load(
    "models/sales_forecasting_model.pkl"
)

print("=" * 70)
print("FUTURE SALES FORECAST")
print("=" * 70)


# =========================================================
# 2. SORT DATA
# =========================================================

df = df.sort_values(
    ["Category", "Region", "Month_Date"]
).reset_index(drop=True)


# =========================================================
# 3. GET LAST 3 MONTHS FOR EACH CATEGORY + REGION
# =========================================================

groups = []

for (category, region), group in df.groupby(
    ["Category", "Region"]
):

    group = group.sort_values("Month_Date")

    if len(group) >= 3:
        groups.append(
            group.tail(3).copy()
        )


# =========================================================
# 4. FORECAST NEXT 3 MONTHS
# =========================================================

future_predictions = []

for group in groups:

    category = group["Category"].iloc[-1]
    region = group["Region"].iloc[-1]

    history = list(
        group["Sales"].values
    )

    last_date = group["Month_Date"].iloc[-1]

    for step in range(1, 4):

        future_date = (
            last_date
            + pd.DateOffset(months=step)
        )

        # -----------------------------------------
        # Lag features
        # -----------------------------------------

        lag_1 = history[-1]
        lag_2 = history[-2]
        lag_3 = history[-3]

        rolling_average = np.mean(
            history[-3:]
        )

        # -----------------------------------------
        # Growth
        # -----------------------------------------

        if lag_2 != 0:
            growth = (
                (lag_1 - lag_2)
                / lag_2
            )
        else:
            growth = 0

        # -----------------------------------------
        # Time features
        # -----------------------------------------

        year = future_date.year
        month = future_date.month
        quarter = future_date.quarter

        # -----------------------------------------
        # Create input
        # -----------------------------------------

        input_data = pd.DataFrame({

            "Sales_Lag_1": [lag_1],

            "Sales_Lag_2": [lag_2],

            "Sales_Lag_3": [lag_3],

            "Rolling_3_Month_Average":
                [rolling_average],

            "Sales_Growth":
                [growth],

            "Year":
                [year],

            "Month":
                [month],

            "Quarter":
                [quarter],

            "Category":
                [category],

            "Region":
                [region]
        })

        # -----------------------------------------
        # Predict
        # -----------------------------------------

        prediction = model.predict(
            input_data
        )[0]

        prediction = max(
            0,
            prediction
        )

        # -----------------------------------------
        # Store result
        # -----------------------------------------

        future_predictions.append({

            "Month_Date":
                future_date,

            "Category":
                category,

            "Region":
                region,

            "Predicted_Sales":
                prediction
        })

        # -----------------------------------------
        # Add prediction to history
        # -----------------------------------------

        history.append(prediction)


# =========================================================
# 5. CREATE FORECAST DATAFRAME
# =========================================================

forecast_df = pd.DataFrame(
    future_predictions
)


# =========================================================
# 6. SAVE FORECAST
# =========================================================

forecast_df.to_csv("data/future_sales_forecast.csv", index=False)


# =========================================================
# 7. DISPLAY RESULTS
# =========================================================

print("\n" + "=" * 70)
print("NEXT 3 MONTHS FORECAST")
print("=" * 70)

print(forecast_df.head(20).to_string(index=False))


# =========================================================
# 8. TOTAL FORECAST BY MONTH
# =========================================================

monthly_forecast = (
    forecast_df.groupby("Month_Date")["Predicted_Sales"].sum().reset_index()
)

print("\n" + "=" * 70)
print("TOTAL EXPECTED SALES BY MONTH")
print("=" * 70)

print(monthly_forecast.to_string(index=False))


print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print("Forecast saved to:")

print("data/future_sales_forecast.csv")