
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import os

project_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

orders_path = os.path.join(
    project_folder, "Data", "olist_orders_dataset.csv"
)

items_path = os.path.join(
    project_folder, "Data", "olist_order_items_dataset.csv"
)

orders = pd.read_csv(orders_path)
items = pd.read_csv(items_path)

print("Datasets loaded successfully!")


print("\nORDERS DATASET")
print(orders.head())
print("Shape:", orders.shape)
orders.info()

print("\nORDER ITEMS DATASET")
print(items.head())
print("Shape:", items.shape)
items.info()



print("Missing values in orders:")
print(orders.isnull().sum())

print("\nMissing values in order items:")
print(items.isnull().sum())


print("Duplicate rows in orders:", orders.duplicated().sum())

print("Duplicate rows in order items:", items.duplicated().sum())


print("Order items numerical statistics:")
print(items[["price", "freight_value"]].describe())

print("\nOrder status distribution:")
print(orders["order_status"].value_counts())

print("\nAverage item price:")
print(items["price"].mean())

print("\nAverage freight value:")
print(items["freight_value"].mean())

print("\nMedian item price:")
print(items["price"].median())

print("\nMedian freight value:")
print(items["freight_value"].median())


# Create a folder for analysis outputs
output_folder = os.path.join(
    os.path.dirname(__file__), "outputs"
)

os.makedirs(output_folder, exist_ok=True)

# Save summary statistics
items[["price", "freight_value"]].describe().to_csv(
    os.path.join(output_folder, "numerical_summary.csv")
)

# Save order status counts
orders["order_status"].value_counts().to_csv(
    os.path.join(output_folder, "order_status_counts.csv")
)

print("\nEDA summary files saved successfully!")


# Convert date columns from text to datetime
date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    orders[column] = pd.to_datetime(
        orders[column], errors="coerce"
    )

print("\nDate columns converted successfully!")

print(orders[date_columns].dtypes)


# Keep only orders that were delivered
delivered_orders = orders[
    orders["order_status"] == "delivered"
].copy()

# Calculate delivery time in days
delivered_orders["delivery_time_days"] = (
    delivered_orders["order_delivered_customer_date"]
    - delivered_orders["order_purchase_timestamp"]
).dt.total_seconds() / (24 * 60 * 60)

# Display delivery time results
print("\nDelivery Time Analysis")
print(
    delivered_orders["delivery_time_days"].describe()
)

print(
    "\nAverage delivery time:",
    round(delivered_orders["delivery_time_days"].mean(), 2),
    "days"
)



# Calculate delivery delay in days
delivered_orders["delay_days"] = (
    delivered_orders["order_delivered_customer_date"]
    - delivered_orders["order_estimated_delivery_date"]
).dt.total_seconds() / (24 * 60 * 60)

# Classify delivery performance
delivered_orders["delivery_performance"] = (
    delivered_orders["delay_days"].apply(
        lambda days: "Late" if days > 0 else "On time or early"
    )
)

# Count on-time and late deliveries
performance_counts = (
    delivered_orders["delivery_performance"].value_counts()
)

print("\nDelivery Performance")
print(performance_counts)

print(
    "\nLate delivery percentage:",
    round(
        (delivered_orders["delay_days"] > 0).mean() * 100,
        2
    ),
    "%"
)


# Calculate monthly order volume
orders["order_month"] = (
    orders["order_purchase_timestamp"].dt.to_period("M")
)

monthly_volume = (
    orders.groupby("order_month")["order_id"]
    .nunique()
    .sort_index()
)

print("\nMonthly Order Volume")
print(monthly_volume)

# Analyze freight charges at item level
print("\nFreight Cost Statistics")
print(items["freight_value"].describe())

# Calculate total freight value for each order
freight_per_order = (
    items.groupby("order_id")["freight_value"].sum()
)

print("\nFreight Per Order Statistics")
print(freight_per_order.describe())

print(
    "\nTotal recorded freight value:",
    round(items["freight_value"].sum(), 2)
)

# STEP 5: VISUALIZATION

# Create output folder for charts
chart_folder = os.path.join(
    os.path.dirname(__file__), "charts"
)
os.makedirs(chart_folder, exist_ok=True)

# Chart 1: Delivery Time Distribution
plt.figure(figsize=(10, 6))

sns.histplot(
    data=delivered_orders,
    x="delivery_time_days",
    bins=30,
    kde=True
)

plt.xlim(0, 60)
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Orders")
plt.tight_layout()

plt.savefig(
    os.path.join(chart_folder, "delivery_time_distribution.png"),
    dpi=300
)
plt.show()
plt.close()


# Chart 2: Monthly Order Volume

# Convert monthly periods to strings for plotting
monthly_volume.index = monthly_volume.index.astype(str)

plt.figure(figsize=(12, 6))

sns.lineplot(
    x=monthly_volume.index,
    y=monthly_volume.values,
    marker="o"
)

plt.title("Monthly Order Volume")
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(chart_folder, "monthly_order_volume.png"),
    dpi=300
)

plt.show()
plt.close()

# Chart 3: Freight Cost Distribution

plt.figure(figsize=(10, 6))

sns.boxplot(
    x=items["freight_value"]
)

plt.title("Distribution of Freight Costs per Item")
plt.xlabel("Freight Value (R$)")

plt.tight_layout()

plt.savefig(
    os.path.join(chart_folder, "freight_cost_boxplot.png"),
    dpi=300
)

plt.show()
plt.close()

# Chart 4: Delivery Time vs Freight Cost

# Calculate total freight for each order
freight_per_order = (
    items.groupby("order_id")["freight_value"].sum()
    .reset_index()
)

# Keep the delivery time and order ID
delivery_freight = delivered_orders[
    ["order_id", "delivery_time_days"]
].merge(
    freight_per_order,
    on="order_id",
    how="inner"
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=delivery_freight,
    x="delivery_time_days",
    y="freight_value",
    alpha=0.3
)

plt.xlim(0, 60)
plt.title("Delivery Time vs. Total Freight Cost")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Total Freight per Order (R$)")
plt.tight_layout()

plt.savefig(
    os.path.join(chart_folder, "delivery_time_vs_freight.png"),
    dpi=300
)

plt.show()
plt.close()

# Chart 5: Correlation Heatmap

# Create a DataFrame with order_id included
correlation_data = delivered_orders[
    ["order_id", "delivery_time_days", "delay_days"]
].copy()

# Calculate total freight cost for each order
freight_per_order = (
    items.groupby("order_id", as_index=False)["freight_value"]
    .sum()
)

# Merge delivery data with freight data
correlation_data = correlation_data.merge(
    freight_per_order,
    on="order_id",
    how="inner"
)

# Select columns for correlation
correlation_matrix = correlation_data[
    ["delivery_time_days", "delay_days", "freight_value"]
].corr()

print("\nCorrelation Matrix")
print(correlation_matrix)

# Plot the heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Heatmap: Logistics Metrics")
plt.tight_layout()

plt.savefig(
    os.path.join(chart_folder, "correlation_heatmap.png"),
    dpi=300
)

plt.show()
plt.close()
