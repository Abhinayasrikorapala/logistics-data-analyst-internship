import pandas as pd

# ============================================================
# 1. LOAD RAW DATA
# ============================================================

file_path = r"C:\Users\KorapalaAbhinaya sri\Desktop\Logistics Data Analyst Internship\Data\olist_orders_dataset.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("WEEK 2 - LOGISTICS DATA PREPROCESSING")
print("=" * 60)

print("\nFirst 5 Rows:")
print(df.head())

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

# ============================================================
# 2. DATA TYPES
# ============================================================

print("\nData Types Before Conversion:")
print(df.dtypes)

# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())

# ============================================================
# 4. DUPLICATE ROWS
# ============================================================

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ============================================================
# 5. ORDER STATUS COUNTS
# ============================================================

print("\nOrder Status Counts:")
print(df["order_status"].value_counts())

# ============================================================
# 6. MISSING VALUES BY ORDER STATUS
# ============================================================

print("\nMissing Delivered Customer Date by Order Status:")
print(
    df[df["order_delivered_customer_date"].isnull()]
    ["order_status"]
    .value_counts()
)

print("\nMissing Approved Date by Order Status:")
print(
    df[df["order_approved_at"].isnull()]
    ["order_status"]
    .value_counts()
)

print("\nMissing Carrier Delivery Date by Order Status:")
print(
    df[df["order_delivered_carrier_date"].isnull()]
    ["order_status"]
    .value_counts()
)

# ============================================================
# 7. CONVERT DATE COLUMNS
# ============================================================

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    df[column] = pd.to_datetime(df[column], errors="coerce")

print("\nData Types After Date Conversion:")
print(df.dtypes)

print("\nMissing Values After Date Conversion:")
print(df.isnull().sum())

# ============================================================
# 8. DATE VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("INVALID DATE CHECKS")
print("=" * 60)

# Delivered before purchase
invalid_delivery = (
    df["order_delivered_customer_date"].notna()
    & (
        df["order_delivered_customer_date"]
        < df["order_purchase_timestamp"]
    )
)

print(
    "Orders delivered before purchase:",
    invalid_delivery.sum()
)

# Customer delivery before carrier delivery
invalid_customer_carrier = (
    df["order_delivered_carrier_date"].notna()
    & df["order_delivered_customer_date"].notna()
    & (
        df["order_delivered_customer_date"]
        < df["order_delivered_carrier_date"]
    )
)

print(
    "Orders delivered to customer before carrier delivery:",
    invalid_customer_carrier.sum()
)

# Approval before purchase
invalid_approval = (
    df["order_approved_at"].notna()
    & (
        df["order_approved_at"]
        < df["order_purchase_timestamp"]
    )
)

print(
    "Orders approved before purchase:",
    invalid_approval.sum()
)

# Carrier delivery before approval
invalid_carrier_approval = (
    df["order_delivered_carrier_date"].notna()
    & df["order_approved_at"].notna()
    & (
        df["order_delivered_carrier_date"]
        < df["order_approved_at"]
    )
)

print(
    "Orders sent to carrier before approval:",
    invalid_carrier_approval.sum()
)

# Late delivery
late_delivery = (
    df["order_delivered_customer_date"].notna()
    & (
        df["order_delivered_customer_date"]
        > df["order_estimated_delivery_date"]
    )
)

print(
    "Orders delivered after estimated delivery date:",
    late_delivery.sum()
)

# ============================================================
# 9. DUPLICATE ORDER IDs
# ============================================================

print("\nDuplicate Order IDs:")
print(df["order_id"].duplicated().sum())

# ============================================================
# 10. MISSING DATES IN DELIVERED ORDERS
# ============================================================

print("\nMissing Dates in Delivered Orders:")

delivered_missing = df[
    (df["order_status"] == "delivered")
    & (
        df[
            [
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date"
            ]
        ].isnull().any(axis=1)
    )
]

print(
    "Delivered orders with missing dates:",
    len(delivered_missing)
)

# ============================================================
# 11. CREATE DELIVERY TIME
# ============================================================

delivered_orders = df[
    df["order_delivered_customer_date"].notna()
].copy()

delivered_orders["delivery_time_days"] = (
    delivered_orders["order_delivered_customer_date"]
    - delivered_orders["order_purchase_timestamp"]
).dt.total_seconds() / (60 * 60 * 24)

print("\nDelivery Time Summary:")
print(delivered_orders["delivery_time_days"].describe())

# ============================================================
# 12. OUTLIER DETECTION USING IQR
# ============================================================

Q1 = delivered_orders["delivery_time_days"].quantile(0.25)
Q3 = delivered_orders["delivery_time_days"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = delivered_orders[
    (delivered_orders["delivery_time_days"] < lower_limit)
    | (delivered_orders["delivery_time_days"] > upper_limit)
]

print("\nDelivery Time Outlier Check:")
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)
print("Number of Delivery Time Outliers:", len(outliers))

# ============================================================
# 13. OUTLIER FLAG
# ============================================================

delivered_orders["delivery_time_outlier"] = (
    (delivered_orders["delivery_time_days"] < lower_limit)
    | (delivered_orders["delivery_time_days"] > upper_limit)
).astype(int)

print("\nOutlier Flag Counts:")
print(
    delivered_orders["delivery_time_outlier"].value_counts()
)

# ============================================================
# 14. DISPLAY LARGEST DELIVERY TIMES
# ============================================================

print("\nLargest Delivery Times:")

print(
    delivered_orders[
        [
            "order_id",
            "order_status",
            "delivery_time_days"
        ]
    ]
    .sort_values(
        "delivery_time_days",
        ascending=False
    )
    .head(10)
)

# ============================================================
# 15. CREATE CLEANED DATASET
# ============================================================

cleaned_df = df.copy()

# Delivery time
cleaned_df["delivery_time_days"] = (
    cleaned_df["order_delivered_customer_date"]
    - cleaned_df["order_purchase_timestamp"]
).dt.total_seconds() / (60 * 60 * 24)

# Delivery delay compared with estimated date
cleaned_df["delivery_delay_days"] = (
    cleaned_df["order_delivered_customer_date"]
    - cleaned_df["order_estimated_delivery_date"]
).dt.total_seconds() / (60 * 60 * 24)

# Late delivery flag
cleaned_df["late_delivery"] = (
    cleaned_df["delivery_delay_days"] > 0
).astype(int)

# Date-quality flags
cleaned_df["invalid_delivery_before_purchase"] = (
    cleaned_df["order_delivered_customer_date"].notna()
    & (
        cleaned_df["order_delivered_customer_date"]
        < cleaned_df["order_purchase_timestamp"]
    )
).astype(int)

cleaned_df["invalid_customer_before_carrier"] = (
    cleaned_df["order_delivered_carrier_date"].notna()
    & cleaned_df["order_delivered_customer_date"].notna()
    & (
        cleaned_df["order_delivered_customer_date"]
        < cleaned_df["order_delivered_carrier_date"]
    )
).astype(int)

cleaned_df["invalid_approval_before_purchase"] = (
    cleaned_df["order_approved_at"].notna()
    & (
        cleaned_df["order_approved_at"]
        < cleaned_df["order_purchase_timestamp"]
    )
).astype(int)

cleaned_df["invalid_carrier_before_approval"] = (
    cleaned_df["order_delivered_carrier_date"].notna()
    & cleaned_df["order_approved_at"].notna()
    & (
        cleaned_df["order_delivered_carrier_date"]
        < cleaned_df["order_approved_at"]
    )
).astype(int)

# ============================================================
# 16. OUTLIER FLAG FOR FULL DATASET
# ============================================================

cleaned_df["delivery_time_outlier"] = (
    cleaned_df["delivery_time_days"] > upper_limit
).fillna(False).astype(int)

# ============================================================
# 17. STANDARDIZATION
# ============================================================

mean_delivery = cleaned_df["delivery_time_days"].mean()
std_delivery = cleaned_df["delivery_time_days"].std()

if std_delivery != 0 and pd.notna(std_delivery):

    cleaned_df["delivery_time_standardized"] = (
        cleaned_df["delivery_time_days"]
        - mean_delivery
    ) / std_delivery

else:

    cleaned_df["delivery_time_standardized"] = 0

# ============================================================
# 18. SAVE CLEANED DATASET
# ============================================================

output_path = r"C:\Users\KorapalaAbhinaya sri\Desktop\Logistics Data Analyst Internship\Data\Week 2 Dataset\cleaned_orders_dataset.csv"

cleaned_df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("CLEANED DATASET SAVED SUCCESSFULLY")
print("=" * 60)

print("Saved to:")
print(output_path)

# ============================================================
# 19. FINAL VERIFICATION
# ============================================================

print("\nFinal Dataset Shape:")
print(cleaned_df.shape)

print("\nFinal Column Names:")
print(cleaned_df.columns)

print("\nFinal Missing Values:")
print(cleaned_df.isnull().sum())

print("\nFinal Duplicate Rows:")
print(cleaned_df.duplicated().sum())

print("\nPreprocessing Completed Successfully!")