import pandas as pd
import numpy as np

def preprocess_data(train_path='train.csv', test_path='test.csv'):
    """
    Preprocess train and test datasets by:
    1. Loading the data
    2. Dropping specified columns
    3. One-hot encoding categorical features
    4. Ensuring train and test have same features
    """

    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Original train shape: {train_df.shape}")
    print(f"Original test shape: {test_df.shape}")

    # Columns to drop
    cols_to_drop = ['Hospital_Id', 'Supplier_Name', 'Hospital_Location', 'Order_Placed_Date', 'Delivery_Date']

    print(f"\nDropping columns: {cols_to_drop}")

    # Drop columns from train
    train_df = train_df.drop(columns=[col for col in cols_to_drop if col in train_df.columns])

    # Drop columns from test (excluding Transport_Cost which shouldn't be in test anyway)
    test_df = test_df.drop(columns=[col for col in cols_to_drop if col in test_df.columns])

    print(f"After dropping columns - Train shape: {train_df.shape}")
    print(f"After dropping columns - Test shape: {test_df.shape}")

    # Separate target variable from train
    if 'Transport_Cost' in train_df.columns:
        y_train = train_df['Transport_Cost'].copy()
        X_train = train_df.drop('Transport_Cost', axis=1)
    else:
        y_train = None
        X_train = train_df.copy()

    X_test = test_df.copy()

    print(f"\nFeatures train shape: {X_train.shape}")
    print(f"Features test shape: {X_test.shape}")

    # Identify categorical columns (object dtype)
    categorical_cols_train = X_train.select_dtypes(include=['object']).columns.tolist()
    categorical_cols_test = X_test.select_dtypes(include=['object']).columns.tolist()

    print(f"\nCategorical columns in train: {categorical_cols_train}")
    print(f"Categorical columns in test: {categorical_cols_test}")

    # Get all categorical columns from both datasets
    all_categorical_cols = list(set(categorical_cols_train + categorical_cols_test))
    print(f"All categorical columns to encode: {all_categorical_cols}")

    # Combine train and test for consistent one-hot encoding
    print("\nCombining datasets for consistent one-hot encoding...")

    # Add a flag to identify train vs test rows
    X_train['is_train'] = 1
    X_test['is_train'] = 0

    # Combine datasets
    combined_df = pd.concat([X_train, X_test], ignore_index=True, sort=False)
    print(f"Combined dataset shape: {combined_df.shape}")

    # One-hot encode categorical variables
    print("\nPerforming one-hot encoding...")

    # Get categorical columns that exist in combined dataset
    categorical_cols = [col for col in all_categorical_cols if col in combined_df.columns]

    if categorical_cols:
        print(f"Encoding columns: {categorical_cols}")

        # One-hot encode
        combined_encoded = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=False, dummy_na=True)

        print(f"Shape after one-hot encoding: {combined_encoded.shape}")

        # Split back into train and test
        train_mask = combined_encoded['is_train'] == 1
        test_mask = combined_encoded['is_train'] == 0

        X_train_encoded = combined_encoded[train_mask].drop('is_train', axis=1)
        X_test_encoded = combined_encoded[test_mask].drop('is_train', axis=1)

    else:
        print("No categorical columns to encode")
        X_train_encoded = X_train.drop('is_train', axis=1)
        X_test_encoded = X_test.drop('is_train', axis=1)

    print(f"\nFinal train features shape: {X_train_encoded.shape}")
    print(f"Final test features shape: {X_test_encoded.shape}")

    # Verify same columns
    train_cols = set(X_train_encoded.columns)
    test_cols = set(X_test_encoded.columns)

    print(f"\nColumns in train but not in test: {train_cols - test_cols}")
    print(f"Columns in test but not in train: {test_cols - train_cols}")
    print(f"Total common columns: {len(train_cols & test_cols)}")

    # Ensure same column order
    common_cols = sorted(list(train_cols & test_cols))
    X_train_final = X_train_encoded[common_cols]
    X_test_final = X_test_encoded[common_cols]

    print(f"\nFinal shapes with same columns:")
    print(f"X_train: {X_train_final.shape}")
    print(f"X_test: {X_test_final.shape}")

    # Print column info
    print(f"\nFinal columns ({len(common_cols)}):")
    for i, col in enumerate(common_cols, 1):
        print(f"{i:3d}. {col}")

    # Print data types
    print(f"\nData types after preprocessing:")
    print("TRAIN:")
    print(X_train_final.dtypes.value_counts())
    print("\nTEST:")
    print(X_test_final.dtypes.value_counts())

    # Check for missing values
    print(f"\nMissing values after preprocessing:")
    print("TRAIN:")
    train_missing = X_train_final.isnull().sum()
    print(f"Total missing: {train_missing.sum()}")
    if train_missing.sum() > 0:
        print("Columns with missing values:")
        print(train_missing[train_missing > 0])

    print("\nTEST:")
    test_missing = X_test_final.isnull().sum()
    print(f"Total missing: {test_missing.sum()}")
    if test_missing.sum() > 0:
        print("Columns with missing values:")
        print(test_missing[test_missing > 0])

    return X_train_final, X_test_final, y_train

# Run the preprocessing
if __name__ == "__main__":
    print("Starting preprocessing...")
    print("="*80)

    X_train, X_test, y_train = preprocess_data()

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)

    if y_train is not None:
        print(f"\nTarget variable (y_train) shape: {y_train.shape}")
        print(f"Target variable stats:")
        print(f"  Mean: {y_train.mean():.2f}")
        print(f"  Std: {y_train.std():.2f}")
        print(f"  Min: {y_train.min():.2f}")
        print(f"  Max: {y_train.max():.2f}")

    # Save processed datasets
    print(f"\nSaving processed datasets...")
    X_train.to_csv('X_train_preprocessed.csv', index=False)
    X_test.to_csv('X_test_preprocessed.csv', index=False)
    if y_train is not None:
        y_train.to_csv('y_train.csv', index=False)

    print("Saved files:")
    print("- X_train_preprocessed.csv")
    print("- X_test_preprocessed.csv")
    if y_train is not None:
        print("- y_train.csv")

    print("\nPreprocessing script completed successfully!")
