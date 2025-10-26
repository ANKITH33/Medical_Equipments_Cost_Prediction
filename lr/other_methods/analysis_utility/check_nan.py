import pandas as pd
import numpy as np

def check_nan_values():
    """
    Check for any NaN values in the preprocessed datasets
    """

    print("Loading preprocessed datasets...")
    X_train = pd.read_csv('X_train_preprocessed.csv')
    X_test = pd.read_csv('X_test_preprocessed.csv')
    y_train = pd.read_csv('y_train.csv')

    print(f"Dataset shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print("="*60)

    # Check X_train for NaN values
    print("CHECKING X_TRAIN FOR NaN VALUES:")
    print("-" * 40)
    X_train_nan = X_train.isnull().sum()
    X_train_total_nan = X_train_nan.sum()

    print(f"Total NaN values in X_train: {X_train_total_nan}")

    if X_train_total_nan > 0:
        print("Columns with NaN values:")
        nan_cols = X_train_nan[X_train_nan > 0]
        for col, count in nan_cols.items():
            percentage = (count / len(X_train)) * 100
            print(f"  {col}: {count} ({percentage:.2f}%)")
    else:
        print("✓ No NaN values found in X_train")

    print()

    # Check X_test for NaN values
    print("CHECKING X_TEST FOR NaN VALUES:")
    print("-" * 40)
    X_test_nan = X_test.isnull().sum()
    X_test_total_nan = X_test_nan.sum()

    print(f"Total NaN values in X_test: {X_test_total_nan}")

    if X_test_total_nan > 0:
        print("Columns with NaN values:")
        nan_cols = X_test_nan[X_test_nan > 0]
        for col, count in nan_cols.items():
            percentage = (count / len(X_test)) * 100
            print(f"  {col}: {count} ({percentage:.2f}%)")
    else:
        print("✓ No NaN values found in X_test")

    print()

    # Check y_train for NaN values
    print("CHECKING Y_TRAIN FOR NaN VALUES:")
    print("-" * 40)

    # Handle case where y_train might be DataFrame or Series
    if isinstance(y_train, pd.DataFrame):
        if y_train.shape[1] == 1:
            y_train_series = y_train.iloc[:, 0]
        else:
            print(f"Warning: y_train has {y_train.shape[1]} columns")
            y_train_series = y_train
    else:
        y_train_series = y_train

    if isinstance(y_train_series, pd.Series):
        y_train_nan_count = y_train_series.isnull().sum()
    else:
        y_train_nan_count = y_train.isnull().sum().sum()

    print(f"Total NaN values in y_train: {y_train_nan_count}")

    if y_train_nan_count > 0:
        if isinstance(y_train, pd.DataFrame):
            nan_cols = y_train.isnull().sum()
            for col, count in nan_cols.items():
                if count > 0:
                    percentage = (count / len(y_train)) * 100
                    print(f"  {col}: {count} ({percentage:.2f}%)")
        else:
            percentage = (y_train_nan_count / len(y_train_series)) * 100
            print(f"  NaN values: {y_train_nan_count} ({percentage:.2f}%)")
    else:
        print("✓ No NaN values found in y_train")

    print()
    print("="*60)

    # Summary
    total_nan_all = X_train_total_nan + X_test_total_nan + y_train_nan_count

    print("SUMMARY:")
    print(f"Total NaN values across all datasets: {total_nan_all}")

    if total_nan_all == 0:
        print("✓ SUCCESS: No NaN values found in any dataset!")
        print("✓ All datasets are ready for modeling.")
    else:
        print("⚠ WARNING: NaN values still exist!")
        print("⚠ Further imputation may be needed.")

    # Additional checks
    print(f"\nADDITIONAL CHECKS:")
    print(f"X_train data types: {X_train.dtypes.value_counts().to_dict()}")
    print(f"X_test data types: {X_test.dtypes.value_counts().to_dict()}")

    # Check if column names match
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)

    if train_cols == test_cols:
        print("✓ X_train and X_test have identical column names")
    else:
        print("⚠ X_train and X_test have different columns:")
        print(f"  In train but not test: {train_cols - test_cols}")
        print(f"  In test but not train: {test_cols - train_cols}")

    print(f"\nColumn count: X_train={len(X_train.columns)}, X_test={len(X_test.columns)}")

if __name__ == "__main__":
    print("CHECKING FOR NaN VALUES IN PREPROCESSED DATASETS")
    print("="*60)
    check_nan_values()
    print("="*60)
    print("NaN CHECK COMPLETE!")
