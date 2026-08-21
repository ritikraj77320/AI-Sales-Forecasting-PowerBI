import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD CLEANED DATA
# ==========================================

df = pd.read_csv("data/cleaned_sales.csv")

print("=" * 60)
print("SALES DATA ANALYSIS")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 2. BASIC STATISTICS
# ==========================================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())


# ==========================================
# 3. TOTAL SALES
# ==========================================

total_sales = df["Sales"].sum()

print("\nTotal Sales:")
print(f"₹{total_sales:,.2f}")


# ==========================================
# 4. TOTAL PROFIT
# ==========================================

total_profit = df["Profit"].sum()

print("\nTotal Profit:")
print(f"₹{total_profit:,.2f}")


# ==========================================
# 5. TOTAL ORDERS
# ==========================================

total_orders = df["Order_ID"].nunique()

print("\nTotal Orders:")
print(total_orders)


# ==========================================
# 6. AVERAGE ORDER VALUE
# ==========================================

average_order_value = df["Sales"].mean()

print("\nAverage Order Value:")
print(f"₹{average_order_value:,.2f}")


# ==========================================
# 7. SALES BY CATEGORY
# ==========================================

category_sales = (df.groupby("Category")["Sales"].sum().sort_values(ascending=False))

print("\n" + "=" * 60)
print("SALES BY CATEGORY")
print("=" * 60)

print(category_sales)


# ==========================================
# 8. SALES BY REGION
# ==========================================

region_sales = (df.groupby("Region")["Sales"].sum().sort_values(ascending=False))

print("\n" + "=" * 60)
print("SALES BY REGION")
print("=" * 60)

print(region_sales)


# ==========================================
# 9. TOP 10 PRODUCTS
# ==========================================

product_sales = (df.groupby("Product")["Sales"].sum().sort_values(ascending=False))

print("\n" + "=" * 60)
print("TOP 10 PRODUCTS")
print("=" * 60)

print(product_sales.head(10))


# ==========================================
# 10. MONTHLY SALES
# ==========================================

monthly_sales = (df.groupby(["Year", "Month"])["Sales"].sum().reset_index())

monthly_sales["Year_Month"] = (monthly_sales["Year"].astype(str) + "-" + monthly_sales["Month"].astype(str).str.zfill(2))

print("\n" + "=" * 60)
print("MONTHLY SALES")
print("=" * 60)

print(monthly_sales.head(20))


# ==========================================
# 11. SALES TREND
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(monthly_sales["Year_Month"],monthly_sales["Sales"])

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# ==========================================
# 12. SALES BY CATEGORY GRAPH
# ==========================================

plt.figure(figsize=(10, 6))

category_sales.plot(kind="bar")

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ==========================================
# 13. SALES BY REGION GRAPH
# ==========================================

plt.figure(figsize=(10, 6))

region_sales.plot(kind="bar")

plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ==========================================
# 14. TOP 10 PRODUCTS GRAPH
# ==========================================

plt.figure(figsize=(12, 6))

product_sales.head(10).plot(kind="bar")

plt.title("Top 10 Products by Sales")
plt.xlabel("Product")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ==========================================
# 15. PROFIT BY CATEGORY
# ==========================================

category_profit = (df.groupby("Category")["Profit"].sum().sort_values(ascending=False))

plt.figure(figsize=(10, 6))

category_profit.plot(kind="bar")

plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ==========================================
# 16. DISCOUNT VS SALES
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(data=df, x="Discount", y="Sales")

plt.title("Discount vs Sales")
plt.xlabel("Discount")
plt.ylabel("Sales")

plt.tight_layout()
plt.show()


# ==========================================
# 17. CORRELATION MATRIX
# ==========================================

numeric_columns = [
    "Quantity",
    "Unit_Price",
    "Discount",
    "Sales",
    "Profit",
    "Profit_Margin"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()