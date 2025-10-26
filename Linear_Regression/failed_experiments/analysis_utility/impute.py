import pandas as pd
import numpy as np

def impute_numerical_features():
    """
    Impute numerical features with median calculated from train set.
    Apply same imputation values to both train and test datasets.
    """

    print("Loading preprocessed data...")
    X_train = pd.read_csv('X_train_preprocessed.csv')
    X_test = pd.read_csv('X_test_preprocessed.csv')
    y_train = pd.read_csv('y_train.csv')

    print(f"Loaded shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")

    # Identify numerical columns (float64 columns)
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nNumerical columns identified: {numerical_cols}")

    # Check missing values before imputation
    print(f"\nMissing values BEFORE imputation:")
    print("TRAIN:")
    train_missing_before = X_train[numerical_cols].isnull().sum()
    for col in numerical_cols:
        if train_missing_before[col] > 0:
            percentage = (train_missing_before[col] / len(X_train)) * 100
            print(f"  {col}: {train_missing_before[col]} ({percentage:.2f}%)")

    print("\nTEST:")
    test_missing_before = X_test[numerical_cols].isnull().sum()
    for col in numerical_cols:
        if test_missing_before[col] > 0:
            percentage = (test_missing_before[col] / len(X_test)) * 100
            print(f"  {col}: {test_missing_before[col]} ({percentage:.2f}%)")

    # Calculate median values from train set for each numerical column
    print(f"\nCalculating median values from train set...")
    median_values = {}

    for col in numerical_cols:
        if train_missing_before[col] > 0:  # Only calculate for columns with missing values
            median_val = X_train[col].median()
            median_values[col] = median_val
            print(f"  {col}: median = {median_val:.4f}")

    # Apply imputation to train set
    print(f"\nImputing train set...")
    X_train_imputed = X_train.copy()
    for col, median_val in median_values.items():
        before_count = X_train_imputed[col].isnull().sum()
        X_train_imputed[col] = X_train_imputed[col].fillna(median_val)
        after_count = X_train_imputed[col].isnull().sum()
        print(f"  {col}: {before_count} -> {after_count} missing values")

    # Apply same imputation values to test set
    print(f"\nImputing test set with same median values...")
    X_test_imputed = X_test.copy()
    for col, median_val in median_values.items():
        if col in X_test_imputed.columns:
            before_count = X_test_imputed[col].isnull().sum()
            X_test_imputed[col] = X_test_imputed[col].fillna(median_val)
            after_count = X_test_imputed[col].isnull().sum()
            print(f"  {col}: {before_count} -> {after_count} missing values")

    # Verify no missing values remain in numerical columns
    print(f"\nMissing values AFTER imputation:")
    print("TRAIN:")
    train_missing_after = X_train_imputed[numerical_cols].isnull().sum()
    total_missing_train = train_missing_after.sum()
    print(f"  Total missing in numerical columns: {total_missing_train}")
    if total_missing_train > 0:
        print("  Remaining missing values:")
        for col in numerical_cols:
            if train_missing_after[col] > 0:
                print(f"    {col}: {train_missing_after[col]}")

    print("\nTEST:")
    test_missing_after = X_test_imputed[numerical_cols].isnull().sum()
    total_missing_test = test_missing_after.sum()
    print(f"  Total missing in numerical columns: {total_missing_test}")
    if total_missing_test > 0:
        print("  Remaining missing values:")
        for col in numerical_cols:
            if test_missing_after[col] > 0:
                print(f"    {col}: {test_missing_after[col]}")

    # Check for any remaining missing values in entire datasets
    print(f"\nTotal missing values in entire datasets after imputation:")
    total_train_missing = X_train_imputed.isnull().sum().sum()
    total_test_missing = X_test_imputed.isnull().sum().sum()
    print(f"Train: {total_train_missing}")
    print(f"Test: {total_test_missing}")

    if total_train_missing > 0:
        print("\nRemaining missing values in train:")
        remaining_missing_train = X_train_imputed.isnull().sum()
        print(remaining_missing_train[remaining_missing_train > 0])

    if total_test_missing > 0:
        print("\nRemaining missing values in test:")
        remaining_missing_test = X_test_imputed.isnull().sum()
        print(remaining_missing_test[remaining_missing_test > 0])

    # Show some statistics of imputed columns
    print(f"\nStatistics of imputed numerical columns (train set):")
    for col in median_values.keys():
        print(f"\n{col}:")
        print(f"  Mean: {X_train_imputed[col].mean():.4f}")
        print(f"  Median: {X_train_imputed[col].median():.4f}")
        print(f"  Std: {X_train_imputed[col].std():.4f}")
        print(f"  Min: {X_train_imputed[col].min():.4f}")
        print(f"  Max: {X_train_imputed[col].max():.4f}")

    return X_train_imputed, X_test_imputed, y_train, median_values

# Run the imputation
if __name__ == "__main__":
    print("Starting numerical imputation...")
    print("="*80)

    X_train_final, X_test_final, y_train, medians_used = impute_numerical_features()

    print("\n" + "="*80)
    print("NUMERICAL IMPUTATION COMPLETE!")
    print("="*80)

    # Save the imputed datasets back to same filenames
    print(f"\nSaving imputed datasets...")
    X_train_final.to_csv('X_train_preprocessed.csv', index=False)
    X_test_final.to_csv('X_test_preprocessed.csv', index=False)
    # y_train doesn't change, but save it anyway to be consistent
    y_train.to_csv('y_train.csv', index=False)

    print("Updated files:")
    print("- X_train_preprocessed.csv (numerical features imputed)")
    print("- X_test_preprocessed.csv (numerical features imputed)")
    print("- y_train.csv")

    print(f"\nImputation summary:")
    print(f"Median values used for imputation:")
    for col, val in medians_used.items():
        print(f"  {col}: {val:.4f}")

    print(f"\nFinal dataset shapes:")
    print(f"X_train: {X_train_final.shape}")
    print(f"X_test: {X_test_final.shape}")

    print("\nNumerical imputation completed successfully!")
    print("All numerical missing values have been filled with train set medians.")
