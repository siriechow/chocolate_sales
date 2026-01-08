# ============================================================
# Project Context & Objective
# ============================================================
"""
Project: Sales Boxes Shipped Classification
Objective:
Clean and preprocess raw sales data to produce a high-quality
dataset suitable for EDA and Machine Learning modeling.

Output:
- cleaned_data.csv (used in EDA & Feature Engineering)
"""

# ============================================================
# Library Imports
# ============================================================
import pandas as pd
import numpy as np

# Display settings for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

# ============================================================
# Data Loading
# ============================================================
# Update file path if needed
df = pd.read_csv("data.csv")

print("✅ Dataset Loaded Successfully")
print("Shape:", df.shape)

# ============================================================
# Initial Data Inspection
# ============================================================
print("\n--- Dataset Info ---")
df.info()

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Sample Records ---")
df.head()

# ============================================================
# Column Name Standardization
# ============================================================
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\n✅ Standardized Column Names:")
print(df.columns)

# ============================================================
# Data Type Corrections
# ============================================================

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Ensure 'boxes_shipped' is integer
df['boxes_shipped'] = df['boxes_shipped'].astype(int)

print("\n✅ Data Types After Correction:")
df.dtypes

# ============================================================
# Cleaning Amount Column
# ============================================================

# Remove currency symbols, commas, and unwanted characters
df['amount'] = (
    df['amount']
    .astype(str)
    .str.replace(r'[^0-9.]', '', regex=True)
)

# Convert to numeric
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

print("\n✅ Amount Column Cleaned")
print(df['amount'].describe())

# ============================================================
# Date Feature Standardization
# ============================================================

print("\n--- Date Range ---")
print("Min Date:", df['date'].min())
print("Max Date:", df['date'].max())

# ============================================================
# Duplicate & Consistency Checks
# ============================================================

# Duplicate rows check
duplicate_count = df.duplicated().sum()
print(f"\nDuplicate Rows Found: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("✅ Duplicates Removed")

# Consistency check for categorical columns
categorical_columns = ['sales_person', 'country', 'product']

for col in categorical_columns:
    df[col] = df[col].str.strip()
    print(f"\nUnique values in {col}: {df[col].nunique()}")

# ============================================================
# Final Sanity Checks
# ============================================================

print("\n--- Final Dataset Info ---")
df.info()

print("\n--- Final Statistical Summary ---")
df.describe(include='all')

# ============================================================
# Export Cleaned Dataset
# ============================================================

output_path = "cleaned_data.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Cleaned dataset successfully saved as '{output_path}'")
