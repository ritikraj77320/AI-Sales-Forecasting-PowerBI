import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD CLEANED DATA
# ==========================================

df = pd.read_csv("data/cleaned_sales.csv")

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

print("Original shape:", df.shape)


# ==========================================
# 2. CONVERT DATE
# ==========================================

df["Order_Date"] = pd.to_datetime(df["Order_Date"])


# ==========================================
# 3. CREATE DATE FEATURES
# ==========================================

df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Quarter"] = df["Order_Date"].dt.quarter
df["Day"] = df["Order_Date"].dt.day
df["Day_of_Week_Number"] = df["Order_Date"].dt.dayofweek


# ==========================================
# 4. CREATE BUSINESS FEATURES
# ==========================================

# Total price before discount
df["Gross_Sales"] = df["Quantity"] * df["Unit_Price"]

# Discount amount
df["Discount_Amount"] = (
    df["Gross_Sales"] * df["Discount"]
)

# Profit margin
df["Profit_Margin"] = np.where(
    df["Sales"] != 0,
    (df["Profit"] / df["Sales"]) * 100,
    0
)


# ==========================================
# 5. CREATE PRICE PER UNIT AFTER DISCOUNT
# ==========================================

df["Net_Unit_Price"] = np.where(
    df["Quantity"] != 0,
    df["Sales"] / df["Quantity"],
    0
)


# ==========================================
# 6. SELECT FEATURES FOR ML
# ==========================================

features = [
    "Quantity",
    "Unit_Price",
    "Discount",
    "Year",
    "Month",
    "Quarter",
    "Day",
    "Day_of_Week_Number",
    "Gross_Sales",
    "Discount_Amount",
    "Net_Unit_Price",
    "Category",
    "Region"
]

target = "Sales"


# ==========================================
# 7. CREATE ML DATASET
# ==========================================

ml_df = df[features + [target]].copy()


# ==========================================
# 8. CHECK MISSING VALUES
# ==========================================

print("\nMissing values:")
print(ml_df.isnull().sum())


# ==========================================
# 9. DISPLAY FEATURES
# ==========================================

print("\nFeatures used by ML model:")

for feature in features:
    print("-", feature)

print("\nTarget variable:")
print("-", target)


# ==========================================
# 10. DISPLAY DATA
# ==========================================

print("\nML Dataset:")
print(ml_df.head())

print("\nML Dataset Shape:")
print(ml_df.shape)


# ==========================================
# 11. SAVE FEATURE-ENGINEERED DATA
# ==========================================

output_file = "data/ml_sales_data.csv"

ml_df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print(f"ML dataset saved to: {output_file}")