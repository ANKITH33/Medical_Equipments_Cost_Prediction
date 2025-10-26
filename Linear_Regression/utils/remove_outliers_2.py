import pandas as pd
import numpy as np

def remove_outliers(X_train_path='X_train_preprocessed.csv', y_train_path='y_train.csv'):
    """
    Remove rows where transport cost is in upper and lower 2.5 percentiles (keeping middle 95%)
    """

    print("Loading preprocessed data...")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    # If y_train is a DataFrame with one column, convert to Series
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    print(f"Original shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    # Calculate percentiles
    lower_percentile = np.percentile(y_train, 2.5)
    upper_percentile = np.percentile(y_train, 97.5)

    print(f"\nTarget variable (Transport_Cost) statistics:")
    print(f"Mean: {y_train.mean():.2f}")
    print(f"Std: {y_train.std():.2f}")
    print(f"Min: {y_train.min():.2f}")
    print(f"Max: {y_train.max():.2f}")

    print(f"\nPercentile thresholds:")
    print(f"2.5th percentile (lower threshold): {lower_percentile:.2f}")
    print(f"97.5th percentile (upper threshold): {upper_percentile:.2f}")

    # Identify outliers
    outlier_mask_lower = y_train <= lower_percentile
    outlier_mask_upper = y_train >= upper_percentile
    outlier_mask = outlier_mask_lower | outlier_mask_upper

    # Count outliers
    num_lower_outliers = outlier_mask_lower.sum()
    num_upper_outliers = outlier_mask_upper.sum()
    total_outliers = outlier_mask.sum()

    print(f"\nOutlier analysis:")
    print(f"Lower outliers (≤ {lower_percentile:.2f}): {num_lower_outliers}")
    print(f"Upper outliers (≥ {upper_percentile:.2f}): {num_upper_outliers}")
    print(f"Total outliers: {total_outliers}")
    print(f"Percentage of data to remove: {(total_outliers/len(y_train))*100:.2f}%")

    # Create mask for data to keep (non-outliers)
    keep_mask = ~outlier_mask

    # Filter both X and y
    X_train_filtered = X_train[keep_mask].copy()
    y_train_filtered = y_train[keep_mask].copy()

    # Reset indices
    X_train_filtered = X_train_filtered.reset_index(drop=True)
    y_train_filtered = y_train_filtered.reset_index(drop=True)

    print(f"\nAfter removing outliers:")
    print(f"X_train shape: {X_train_filtered.shape}")
    print(f"y_train shape: {y_train_filtered.shape}")
    print(f"Rows removed: {len(y_train) - len(y_train_filtered)}")
    print(f"Rows remaining: {len(y_train_filtered)}")

    # New target variable statistics
    print(f"\nNew target variable statistics (after outlier removal):")
    print(f"Mean: {y_train_filtered.mean():.2f}")
    print(f"Std: {y_train_filtered.std():.2f}")
    print(f"Min: {y_train_filtered.min():.2f}")
    print(f"Max: {y_train_filtered.max():.2f}")
    print(f"Skewness: {y_train_filtered.skew():.2f}")

    # Show some percentiles of the cleaned data
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles of cleaned target variable:")
    for p in percentiles:
        val = np.percentile(y_train_filtered, p)
        print(f"{p:2d}th percentile: {val:.2f}")

    # Check missing values in filtered data
    print(f"\nMissing values in filtered X_train:")
    missing_values = X_train_filtered.isnull().sum()
    total_missing = missing_values.sum()
    print(f"Total missing values: {total_missing}")

    if total_missing > 0:
        print("Columns with missing values:")
        missing_cols = missing_values[missing_values > 0]
        for col, count in missing_cols.items():
            percentage = (count / len(X_train_filtered)) * 100
            print(f"  {col}: {count} ({percentage:.2f}%)")

    return X_train_filtered, y_train_filtered

# Run the outlier removal
if __name__ == "__main__":
    print("Starting outlier removal...")
    print("="*80)

    X_train_clean, y_train_clean = remove_outliers()

    print("\n" + "="*80)
    print("OUTLIER REMOVAL COMPLETE!")
    print("="*80)

    # Save the cleaned datasets back to the same preprocessed filenames
    print(f"\nOverwriting preprocessed datasets with outlier-free data...")
    X_train_clean.to_csv('X_train_preprocessed.csv', index=False)
    y_train_clean.to_csv('y_train.csv', index=False)

    print("Updated files:")
    print("- X_train_preprocessed.csv (outliers removed)")
    print("- y_train.csv (outliers removed)")

    print(f"\nData reduction summary:")
    print(f"Original samples: 5000")
    print(f"Samples after outlier removal: {len(y_train_clean)}")
    print(f"Reduction: {5000 - len(y_train_clean)} samples ({((5000 - len(y_train_clean))/5000)*100:.2f}%)")

    print("\nOutlier removal completed successfully!")
    print("Preprocessed files have been updated with cleaned data.")
