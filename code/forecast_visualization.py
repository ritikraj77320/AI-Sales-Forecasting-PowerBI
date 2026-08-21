import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. LOAD HISTORICAL DATA
# =========================================================

historical = pd.read_csv("data/cleaned_sales.csv")

historical["Order_Date"] = pd.to_datetime(
    historical["Order_Date"]
)


# =========================================================
# 2. AGGREGATE HISTORICAL SALES BY MONTH
# =========================================================

historical["Month_Date"] = (
    historical["Order_Date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

monthly_historical = (
    historical
    .groupby("Month_Date")["Sales"]
    .sum()
    .reset_index()
)

monthly_historical.rename(
    columns={"Sales": "Actual_Sales"},
    inplace=True
)


# =========================================================
# 3. LOAD FUTURE FORECAST
# =========================================================

forecast = pd.read_csv("data/future_sales_forecast.csv")

forecast["Month_Date"] = pd.to_datetime(
    forecast["Month_Date"]
)


# =========================================================
# 4. AGGREGATE FORECAST BY MONTH
# =========================================================

monthly_forecast = (
    forecast
    .groupby("Month_Date")["Predicted_Sales"]
    .sum()
    .reset_index()
)


# =========================================================
# 5. DISPLAY FORECAST
# =========================================================

print("=" * 70)
print("FUTURE SALES FORECAST")
print("=" * 70)

print(monthly_forecast.to_string(index=False))


# =========================================================
# 6. HISTORICAL VS FORECAST GRAPH
# =========================================================

plt.figure(figsize=(14, 7))

plt.plot(
    monthly_historical["Month_Date"],
    monthly_historical["Actual_Sales"],
    marker="o",
    label="Historical Sales"
)

plt.plot(
    monthly_forecast["Month_Date"],
    monthly_forecast["Predicted_Sales"],
    marker="o",
    linestyle="--",
    label="Forecast Sales"
)

plt.title("Historical Sales vs Future Sales Forecast")

plt.xlabel("Month")
plt.ylabel("Sales")

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# =========================================================
# 7. FUTURE FORECAST ONLY
# =========================================================

plt.figure(figsize=(10, 6))

plt.bar(
    monthly_forecast["Month_Date"].dt.strftime("%Y-%m"),
    monthly_forecast["Predicted_Sales"]
)

plt.title(
    "Next 3 Months Sales Forecast"
)

plt.xlabel("Month")

plt.ylabel("Predicted Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# =========================================================
# 8. FORECAST BY CATEGORY
# =========================================================

category_forecast = (
    forecast
    .groupby("Category")["Predicted_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 70)
print("FORECAST SALES BY CATEGORY")
print("=" * 70)

print(category_forecast.to_string())


plt.figure(figsize=(10, 6))

category_forecast.plot(
    kind="bar"
)

plt.title("Future Sales Forecast by Category")

plt.xlabel("Category")

plt.ylabel("Predicted Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# =========================================================
# 9. FORECAST BY REGION
# =========================================================

region_forecast = (
    forecast
    .groupby("Region")["Predicted_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 70)
print("FORECAST SALES BY REGION")
print("=" * 70)

print(region_forecast.to_string())


plt.figure(figsize=(10, 6))

region_forecast.plot(kind="bar")

plt.title("Future Sales Forecast by Region")

plt.xlabel("Region")

plt.ylabel("Predicted Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# =========================================================
# 10. MONTH-TO-MONTH FORECAST CHANGE
# =========================================================

monthly_forecast["Sales_Change"] = (
    monthly_forecast["Predicted_Sales"].diff()
)

monthly_forecast["Growth_Percentage"] = (
    monthly_forecast["Predicted_Sales"]
    .pct_change()
    * 100
)


print("\n" + "=" * 70)
print("FORECAST GROWTH ANALYSIS")
print("=" * 70)

print(monthly_forecast.to_string(index=False))


# =========================================================
# 11. SAVE DASHBOARD DATA
# =========================================================

monthly_forecast.to_csv(
    "data/monthly_forecast_dashboard.csv",
    index=False
)

category_forecast.reset_index().to_csv(
    "data/category_forecast_dashboard.csv",
    index=False
)

region_forecast.reset_index().to_csv(
    "data/region_forecast_dashboard.csv",
    index=False
)


# =========================================================
# 12. SUCCESS
# =========================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print("Dashboard files created:")
print("1. data/monthly_forecast_dashboard.csv")
print("2. data/category_forecast_dashboard.csv")
print("3. data/region_forecast_dashboard.csv")