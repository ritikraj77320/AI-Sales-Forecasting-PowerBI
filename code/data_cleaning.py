import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/sales_data_12000.csv")

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)

print("Rows and columns:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 2. CHECK DATA TYPES
# ==========================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())


# ==========================================
# 4. CHECK DUPLICATES
# ==========================================

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

print("Duplicate rows:", df.duplicated().sum())


# ==========================================
# 5. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("\nAfter removing duplicates:", df.shape)


# ==========================================
# 6. CONVERT DATE COLUMN
# ==========================================

df["Order_Date"] = pd.to_datetime(df["Order_Date"],errors="coerce")

print("\nOrder_Date data type:")
print(df["Order_Date"].dtype)


# ==========================================
# 7. HANDLE MISSING VALUES
# ==========================================

# Fill missing discount with 0
df["Discount"] = df["Discount"].fillna(0)

# Fill missing region with the most common region
df["Region"] = df["Region"].fillna(df["Region"].mode()[0])


# ==========================================
# 8. CHECK MISSING VALUES AGAIN
# ==========================================

print("\n" + "=" * 60)
print("MISSING VALUES AFTER CLEANING")
print("=" * 60)

print(df.isnull().sum())


# ==========================================
# 9. CHECK INVALID VALUES
# ==========================================

print("\n" + "=" * 60)
print("INVALID VALUES")
print("=" * 60)

print("Negative Quantity:",(df["Quantity"] < 0).sum())

print("Negative Unit Price:",(df["Unit_Price"] < 0).sum())

print("Negative Sales:",(df["Sales"] < 0).sum())


# ==========================================
# 10. REMOVE INVALID VALUES
# ==========================================

df = df[df["Quantity"] > 0]
df = df[df["Unit_Price"] > 0]
df = df[df["Sales"] >= 0]


# ==========================================
# 11. CREATE NEW FEATURES
# ==========================================

df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Quarter"] = df["Order_Date"].dt.quarter
df["Day"] = df["Order_Date"].dt.day
df["Day_of_Week"] = df["Order_Date"].dt.day_name()

df["Profit_Margin"] = (df["Profit"] / df["Sales"] * 100)

# Avoid infinite values
df["Profit_Margin"] = df["Profit_Margin"].replace([np.inf, -np.inf],np.nan)

df["Profit_Margin"] = df["Profit_Margin"].fillna(0)


# ==========================================
# 12. FINAL DATASET INFORMATION
# ==========================================

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print("Rows and columns:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 cleaned rows:")
print(df.head())


# ==========================================
# 13. SAVE CLEANED DATA
# ==========================================

output_file = "data/cleaned_sales.csv"

df.to_csv(output_file,index=False)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print(f"Cleaned dataset saved to: {output_file}")