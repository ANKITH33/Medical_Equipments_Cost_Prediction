import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
print("Loading datasets...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Train dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")
print("="*80)

# Basic info about train dataset
print("TRAIN DATASET INFO:")
print("="*50)
print(train_df.info())
print("\n")

print("TRAIN DATASET COLUMNS:")
print("="*50)
for i, col in enumerate(train_df.columns):
    print(f"{i+1:2d}. {col}")
print("\n")

print("TRAIN DATASET HEAD:")
print("="*50)
print(train_df.head())
print("\n")

print("TRAIN DATASET DESCRIBE (NUMERIC):")
print("="*50)
print(train_df.describe())
print("\n")

print("TRAIN DATASET DESCRIBE (ALL):")
print("="*50)
print(train_df.describe(include='all'))
print("\n")

# Check data types
print("TRAIN DATASET DATA TYPES:")
print("="*50)
print(train_df.dtypes)
print("\n")

# Check for missing values
print("TRAIN DATASET MISSING VALUES:")
print("="*50)
missing_train = train_df.isnull().sum()
missing_percent_train = (missing_train / len(train_df)) * 100
missing_df_train = pd.DataFrame({
    'Column': missing_train.index,
    'Missing_Count': missing_train.values,
    'Missing_Percentage': missing_percent_train.values
}).sort_values('Missing_Count', ascending=False)
print(missing_df_train)
print("\n")

# Check unique values for each column
print("TRAIN DATASET UNIQUE VALUES COUNT:")
print("="*50)
for col in train_df.columns:
    unique_count = train_df[col].nunique()
    print(f"{col:30s}: {unique_count:6d} unique values")
print("\n")

# Show unique values for categorical columns (with fewer unique values)
print("TRAIN DATASET UNIQUE VALUES (for columns with ≤ 20 unique values):")
print("="*70)
for col in train_df.columns:
    unique_count = train_df[col].nunique()
    if unique_count <= 20:
        print(f"\n{col} ({unique_count} unique values):")
        print(train_df[col].value_counts().head(20))
print("\n")

# Basic info about test dataset
print("TEST DATASET INFO:")
print("="*50)
print(test_df.info())
print("\n")

print("TEST DATASET COLUMNS:")
print("="*50)
for i, col in enumerate(test_df.columns):
    print(f"{i+1:2d}. {col}")
print("\n")

print("TEST DATASET HEAD:")
print("="*50)
print(test_df.head())
print("\n")

print("TEST DATASET DESCRIBE (NUMERIC):")
print("="*50)
print(test_df.describe())
print("\n")

print("TEST DATASET DESCRIBE (ALL):")
print("="*50)
print(test_df.describe(include='all'))
print("\n")

# Check data types
print("TEST DATASET DATA TYPES:")
print("="*50)
print(test_df.dtypes)
print("\n")

# Check for missing values in test
print("TEST DATASET MISSING VALUES:")
print("="*50)
missing_test = test_df.isnull().sum()
missing_percent_test = (missing_test / len(test_df)) * 100
missing_df_test = pd.DataFrame({
    'Column': missing_test.index,
    'Missing_Count': missing_test.values,
    'Missing_Percentage': missing_percent_test.values
}).sort_values('Missing_Count', ascending=False)
print(missing_df_test)
print("\n")

# Check unique values for each column in test
print("TEST DATASET UNIQUE VALUES COUNT:")
print("="*50)
for col in test_df.columns:
    unique_count = test_df[col].nunique()
    print(f"{col:30s}: {unique_count:6d} unique values")
print("\n")

# Compare columns between train and test
print("COLUMN COMPARISON BETWEEN TRAIN AND TEST:")
print("="*50)
train_cols = set(train_df.columns)
test_cols = set(test_df.columns)

print(f"Columns in train but not in test: {train_cols - test_cols}")
print(f"Columns in test but not in train: {test_cols - train_cols}")
print(f"Common columns: {len(train_cols & test_cols)}")
print("\n")

# Check target variable if it exists in train
if 'transport_cost' in train_df.columns:
    print("TARGET VARIABLE (transport_cost) ANALYSIS:")
    print("="*50)
    print(f"Mean: {train_df['transport_cost'].mean():.2f}")
    print(f"Median: {train_df['transport_cost'].median():.2f}")
    print(f"Std: {train_df['transport_cost'].std():.2f}")
    print(f"Min: {train_df['transport_cost'].min():.2f}")
    print(f"Max: {train_df['transport_cost'].max():.2f}")
    print(f"Skewness: {train_df['transport_cost'].skew():.2f}")
    print(f"Kurtosis: {train_df['transport_cost'].kurtosis():.2f}")

    # Percentiles for outlier detection
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        val = np.percentile(train_df['transport_cost'].dropna(), p)
        print(f"{p:2d}th percentile: {val:.2f}")
    print("\n")

# Attempt to infer and convert data types
print("ATTEMPTING TO CONVERT DATA TYPES...")
print("="*50)

def convert_datatypes(df, dataset_name):
    df_converted = df.copy()
    conversions = {}

    for col in df_converted.columns:
        original_dtype = df_converted[col].dtype

        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(df_converted[col]):
            continue

        # Try to convert to numeric
        try:
            # First, try direct conversion
            converted = pd.to_numeric(df_converted[col], errors='coerce')

            # If we get some numeric values, it might be worth converting
            if not converted.isna().all():
                # Check if it's mostly integers
                non_null_converted = converted.dropna()
                if len(non_null_converted) > 0:
                    if all(non_null_converted == non_null_converted.astype(int)):
                        df_converted[col] = converted.astype('Int64')  # Nullable integer
                        conversions[col] = f"{original_dtype} -> Int64"
                    else:
                        df_converted[col] = converted
                        conversions[col] = f"{original_dtype} -> float64"
        except:
            pass

    print(f"\n{dataset_name} DATA TYPE CONVERSIONS:")
    for col, conversion in conversions.items():
        print(f"{col:30s}: {conversion}")

    return df_converted

# Convert data types
train_converted = convert_datatypes(train_df, "TRAIN")
test_converted = convert_datatypes(test_df, "TEST")

print("\nFINAL DATA TYPES AFTER CONVERSION:")
print("="*50)
print("TRAIN:")
print(train_converted.dtypes)
print("\nTEST:")
print(test_converted.dtypes)

print("\nEXPLORATORY ANALYSIS COMPLETE!")
print("="*50)
