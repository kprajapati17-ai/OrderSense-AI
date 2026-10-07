import pandas as pd

# Load dataset
df = pd.read_csv("part1_return_prediction/data/orders_dataset.csv")

# First 5 rows
print("\n--- First 5 Rows ---")
print(df.head())

# Number of rows and columns
print("\n--- Dataset Shape ---")
print(df.shape)

# Column names
print("\n--- Column Names ---")
print(df.columns.tolist())

# Data types
print("\n--- Data Types ---")
print(df.dtypes)

# Missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Statistical summary
print("\n--- Statistical Summary ---")
print(df.describe())

# Duplicate rows
print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# Dataset information
print("\n--- Dataset Info ---")
print(df.info())