import pandas as pd
import numpy as np

# =========================================================
# 1. LOAD CLEANED DATA
# =========================================================

df = pd.read_csv("data/cleaned_sales.csv")

print("=" * 60)
print("SALES FORECASTING DATASET")
print("=" * 60)

print("Original dataset shape:", df.shape)


# =========================================================
# 2. CONVERT DATE
# =========================================================

df["Order_Date"] = pd.to_datetime(df["Order_Date"])


# =========================================================
# 3. CREATE MONTH COLUMN
# =========================================================

df["Month_Date"] = (
    df["Order_Date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)


# =========================================================
# 4. AGGREGATE SALES MONTHLY
# =========================================================

monthly_sales = (
    df.groupby(
        [
            "Month_Date",
            "Category",
            "Region"
        ]
    )["Sales"]
    .sum()
    .reset_index()
)


print("\nMonthly dataset shape:")
print(monthly_sales.shape)


# =========================================================
# 5. SORT DATA
# =========================================================

monthly_sales = monthly_sales.sort_values(
    [
        "Category",
        "Region",
        "Month_Date"
    ]
).reset_index(drop=True)


# =========================================================
# 6. CREATE TIME FEATURES
# =========================================================

monthly_sales["Year"] = (
    monthly_sales["Month_Date"].dt.year
)

monthly_sales["Month"] = (
    monthly_sales["Month_Date"].dt.month
)

monthly_sales["Quarter"] = (
    monthly_sales["Month_Date"].dt.quarter
)


# =========================================================
# 7. CREATE LAG FEATURES
# =========================================================

# Previous month sales
monthly_sales["Sales_Lag_1"] = (
    monthly_sales
    .groupby(["Category", "Region"])["Sales"]
    .shift(1)
)

# Sales two months ago
monthly_sales["Sales_Lag_2"] = (
    monthly_sales
    .groupby(["Category", "Region"])["Sales"]
    .shift(2)
)

# Sales three months ago
monthly_sales["Sales_Lag_3"] = (
    monthly_sales
    .groupby(["Category", "Region"])["Sales"]
    .shift(3)
)


# =========================================================
# 8. CREATE ROLLING FEATURES
# =========================================================

monthly_sales["Rolling_3_Month_Average"] = (
    monthly_sales
    .groupby(["Category", "Region"])["Sales"]
    .transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
)


# =========================================================
# 9. CREATE SALES GROWTH
# =========================================================

monthly_sales["Sales_Growth"] = (
    monthly_sales["Sales_Lag_1"]
    .pct_change()
)


# =========================================================
# 10. REMOVE MISSING VALUES
# =========================================================

forecast_df = monthly_sales.dropna().copy()


# =========================================================
# 11. HANDLE INFINITE VALUES
# =========================================================

forecast_df = forecast_df.replace(
    [np.inf, -np.inf],
    np.nan
)

forecast_df = forecast_df.dropna()


# =========================================================
# 12. DISPLAY DATA
# =========================================================

print("\n" + "=" * 60)
print("FORECASTING FEATURES")
print("=" * 60)

print(
    forecast_df[
        [
            "Month_Date",
            "Category",
            "Region",
            "Sales",
            "Sales_Lag_1",
            "Sales_Lag_2",
            "Sales_Lag_3",
            "Rolling_3_Month_Average",
            "Sales_Growth"
        ]
    ].head(15)
)


# =========================================================
# 13. DISPLAY COLUMNS
# =========================================================

print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)

print(forecast_df.columns.tolist())


# =========================================================
# 14. SAVE FORECASTING DATASET
# =========================================================

output_file = "data/forecasting_sales_data.csv"

forecast_df.to_csv(
    output_file,
    index=False
)


# =========================================================
# 15. FINAL INFORMATION
# =========================================================

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print("Forecasting dataset saved to:", output_file)

print("Final dataset shape:", forecast_df.shape)