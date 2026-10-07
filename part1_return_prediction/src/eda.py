import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
df = pd.read_csv("part1_return_prediction/data/orders_dataset.csv")

print("\n--- Return Distribution ---")
print(df["returned"].value_counts())

print("\n--- Return Percentage ---")
print(df["returned"].value_counts(normalize=True) * 100)

print("\n--- Product Category ---")
print(df["product_category"].value_counts())

print("\n--- Payment Method ---")
print(df["payment_method"].value_counts())

print("\n--- Previous Returns ---")
print(df["num_previous_returns"].describe())

print("\n--- Delivery Days ---")
print(df["delivery_days"].describe())

print("\n--- Price ---")
print(df["price_inr"].describe())


## Return rate based on previous returns
return_rate = df.groupby("num_previous_returns")["returned"].mean() * 100

print("\n--- Return Rate by Previous Returns ---")
print(return_rate)

# Previous returns count and return rate
previous_return_analysis = df.groupby("num_previous_returns")["returned"].agg(
    ["count", "mean"]
)

previous_return_analysis["return_rate_pct"] = (
    previous_return_analysis["mean"] * 100
)

print("\n--- Previous Returns Analysis ---")
print(previous_return_analysis)

# Return rate by delivery days
delivery_analysis = df.groupby("delivery_days")["returned"].agg(
    ["count", "mean"]
)

delivery_analysis["return_rate_pct"] = (
    delivery_analysis["mean"] * 100
)

print("\n--- Delivery Days Analysis ---")
print(delivery_analysis)