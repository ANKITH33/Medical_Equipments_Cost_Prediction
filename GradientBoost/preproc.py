import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(123)


def create_engineered_features(dataframe):
    """
    Generate new features including corrected temporal features and physical dimensions.
    """
    df_copy = dataframe.copy()

    # --- Date Processing and Correction ---
    # Parse dates with error handling
    df_copy['OrderDate_parsed'] = pd.to_datetime(df_copy['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
    df_copy['DeliveryDate_parsed'] = pd.to_datetime(df_copy['Delivery_Date'], format='%m/%d/%y', errors='coerce')

    # Calculate initial time difference
    initial_duration = (df_copy['DeliveryDate_parsed'] - df_copy['OrderDate_parsed']).dt.days

    # Fix invalid dates where delivery appears before order
    invalid_dates_mask = initial_duration < 0

    # Swap dates for invalid entries
    if invalid_dates_mask.any():
        print(f"    Found {invalid_dates_mask.sum()} invalid date pairs (delivery before order) - correcting...")
        df_copy.loc[invalid_dates_mask, ['OrderDate_parsed', 'DeliveryDate_parsed']] = \
            df_copy.loc[invalid_dates_mask, ['DeliveryDate_parsed', 'OrderDate_parsed']].values

    # --- Create Temporal Features from Corrected Dates ---
    df_copy['Days_to_Delivery'] = (df_copy['DeliveryDate_parsed'] - df_copy['OrderDate_parsed']).dt.days

    # Extract date components from corrected order date
    df_copy['Order_Month_Num'] = df_copy['OrderDate_parsed'].dt.month.fillna(0).astype(int)
    df_copy['Order_Year_Num'] = df_copy['OrderDate_parsed'].dt.year.fillna(0).astype(int)
    df_copy['Order_Weekday'] = df_copy['OrderDate_parsed'].dt.dayofweek.fillna(-1).astype(int)

    # --- Physical Dimension Features ---
    # Use temporary fills for volume calculation
    height_temp = df_copy['Equipment_Height'].fillna(1.0)
    width_temp = df_copy['Equipment_Width'].fillna(1.0)

    # Create volume proxy feature
    df_copy['Equipment_Area'] = height_temp * width_temp

    # Remove temporary datetime columns
    df_copy = df_copy.drop(columns=['OrderDate_parsed', 'DeliveryDate_parsed'])

    # Remove original dimension columns after creating derived features
    df_copy = df_copy.drop(columns=['Equipment_Height', 'Equipment_Width'])

    return df_copy

def execute_preprocessing(training_data, testing_data):
    """
    Complete preprocessing pipeline with feature engineering and encoding.
    Enhanced with comprehensive debugging and validation.
    """
    print("Initiating preprocessing pipeline...")

    # Print initial dataset info
    print(f"Initial dataset shapes:")
    print(f"  Training data: {training_data.shape}")
    print(f"  Testing data: {testing_data.shape}")

    print(f"\nInitial column comparison:")
    train_initial_cols = set(training_data.columns)
    test_initial_cols = set(testing_data.columns)
    print(f"  Columns in train but not in test: {train_initial_cols - test_initial_cols}")
    print(f"  Columns in test but not in train: {test_initial_cols - train_initial_cols}")
    print(f"  Common columns: {len(train_initial_cols & test_initial_cols)}")

    # --- Target Variable Processing ---
    print(f"\nProcessing target variable...")
    print(f"  Original target shape: {training_data['Transport_Cost'].shape}")
    print(f"  Original target stats:")
    print(f"    Mean: {training_data['Transport_Cost'].mean():.2f}")
    print(f"    Std: {training_data['Transport_Cost'].std():.2f}")
    print(f"    Min: {training_data['Transport_Cost'].min():.2f}")
    print(f"    Max: {training_data['Transport_Cost'].max():.2f}")

    # Remove non-positive transport costs
    positive_mask = training_data['Transport_Cost'] > 0
    negative_count = (~positive_mask).sum()
    if negative_count > 0:
        print(f"    Removing {negative_count} non-positive transport cost entries")

    training_data = training_data[positive_mask].copy()
    target_variable = np.log1p(training_data['Transport_Cost'])
    training_data = training_data.drop(columns=['Transport_Cost'])

    print(f"  After filtering - Training shape: {training_data.shape}")
    print(f"  Log-transformed target stats:")
    print(f"    Mean: {target_variable.mean():.4f}")
    print(f"    Std: {target_variable.std():.4f}")
    print(f"    Min: {target_variable.min():.4f}")
    print(f"    Max: {target_variable.max():.4f}")

    # --- Preserve Test Identifiers ---
    test_hospital_ids = testing_data['Hospital_Id']
    print(f"\nPreserved {len(test_hospital_ids)} test hospital IDs")

    # --- Check Missing Values Before Imputation ---
    print(f"\nMissing values BEFORE imputation:")
    print("TRAINING:")
    train_missing_before = training_data.isnull().sum()
    train_total_missing = train_missing_before.sum()
    print(f"  Total missing: {train_total_missing}")
    if train_total_missing > 0:
        missing_cols = train_missing_before[train_missing_before > 0]
        for col, count in missing_cols.items():
            percentage = (count / len(training_data)) * 100
            print(f"    {col}: {count} ({percentage:.2f}%)")

    print("TESTING:")
    test_missing_before = testing_data.isnull().sum()
    test_total_missing = test_missing_before.sum()
    print(f"  Total missing: {test_total_missing}")
    if test_total_missing > 0:
        missing_cols = test_missing_before[test_missing_before > 0]
        for col, count in missing_cols.items():
            percentage = (count / len(testing_data)) * 100
            print(f"    {col}: {count} ({percentage:.2f}%)")

    # --- Missing Value Imputation Strategy ---
    numerical_missing_features = ['Supplier_Reliability', 'Equipment_Height', 'Equipment_Width', 'Equipment_Weight', 'Equipment_Value']
    categorical_missing_features = ['Transport_Method', 'Equipment_Type', 'Rural_Hospital']

    print("\nComputing imputation statistics from training data exclusively...")
    fill_values = {}

    # Calculate medians for numerical features
    for feature in numerical_missing_features:
        if feature in training_data.columns:
            fill_values[feature] = training_data[feature].median()
            print(f"  {feature}: median = {fill_values[feature]:.4f}")

    # Calculate modes for categorical features
    for feature in categorical_missing_features:
        if feature in training_data.columns:
            mode_result = training_data[feature].mode()
            fill_values[feature] = mode_result[0] if not mode_result.empty else 'Missing'
            print(f"  {feature}: mode = '{fill_values[feature]}'")

    print("Applying computed fill values to both datasets...")
    for feature, fill_value in fill_values.items():
        if feature in training_data.columns:
            before_train = training_data[feature].isnull().sum()
            training_data[feature] = training_data[feature].fillna(fill_value)
            after_train = training_data[feature].isnull().sum()
            print(f"  Train {feature}: {before_train} -> {after_train} missing")

        if feature in testing_data.columns:
            before_test = testing_data[feature].isnull().sum()
            testing_data[feature] = testing_data[feature].fillna(fill_value)
            after_test = testing_data[feature].isnull().sum()
            print(f"  Test {feature}: {before_test} -> {after_test} missing")

    # --- Feature Engineering Application ---
    print("\nExecuting feature engineering transformations...")
    training_data = create_engineered_features(training_data)
    testing_data = create_engineered_features(testing_data)

    print(f"  After feature engineering:")
    print(f"    Training shape: {training_data.shape}")
    print(f"    Testing shape: {testing_data.shape}")

    # --- Dataset Combination for Consistent Encoding ---
    print("Combining datasets for consistent encoding...")

    # Add a flag to identify train vs test rows
    training_data['is_train'] = 1
    testing_data['is_train'] = 0

    combined_dataset = pd.concat([training_data, testing_data], axis=0, ignore_index=True, sort=False)
    print(f"  Combined dataset shape: {combined_dataset.shape}")

    # --- Column Removal ---
    columns_to_remove = [
        'Hospital_Id', 'Supplier_Name', 'Hospital_Location', 'Supplier_Reliability',
        'Order_Placed_Date', 'Delivery_Date'
    ]

    columns_actually_removed = [col for col in columns_to_remove if col in combined_dataset.columns]
    print(f"Removing columns: {columns_actually_removed}")

    combined_dataset = combined_dataset.drop(columns=columns_actually_removed, errors='ignore')
    print(f"  After column removal: {combined_dataset.shape}")

    # --- Data Type Analysis ---
    print(f"\nData types before encoding:")
    dtype_counts = combined_dataset.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")

    # --- Categorical Variable Encoding ---
    print("Applying categorical encoding...")

    # Binary encoding
    binary_encoding_map = {'Yes': 1, 'No': 0}
    if 'Rural_Hospital' in combined_dataset.columns:
        before_unique = combined_dataset['Rural_Hospital'].nunique()
        combined_dataset['Rural_Hospital'] = combined_dataset['Rural_Hospital'].map(binary_encoding_map)
        print(f"  Rural_Hospital: {before_unique} unique -> binary encoded")

    binary_feature_list = ['CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service', 'Fragile_Equipment']
    for feature in binary_feature_list:
        if feature in combined_dataset.columns:
            before_unique = combined_dataset[feature].nunique()
            combined_dataset[feature] = combined_dataset[feature].map(binary_encoding_map)
            print(f"  {feature}: {before_unique} unique -> binary encoded")

    # Identify categorical columns for one-hot encoding
    categorical_features_for_ohe = ['Equipment_Type', 'Transport_Method', 'Hospital_Info', 'Order_Year_Num']
    existing_categorical = [col for col in categorical_features_for_ohe if col in combined_dataset.columns]

    print(f"One-hot encoding columns: {existing_categorical}")

    if existing_categorical:
        # Show unique values before encoding
        for col in existing_categorical:
            unique_count = combined_dataset[col].nunique()
            print(f"  {col}: {unique_count} unique values")

        # One-hot encode
        combined_dataset = pd.get_dummies(combined_dataset, columns=existing_categorical, drop_first=True, dtype=int)
        print(f"  After one-hot encoding: {combined_dataset.shape}")

    # --- Final Dataset Split ---
    print("Performing final dataset separation...")

    # Split back using the flag
    train_mask = combined_dataset['is_train'] == 1
    test_mask = combined_dataset['is_train'] == 0

    X_training = combined_dataset[train_mask].drop('is_train', axis=1).copy()
    X_testing = combined_dataset[test_mask].drop('is_train', axis=1).copy()

    # Reset indices
    X_training = X_training.reset_index(drop=True)
    X_testing = X_testing.reset_index(drop=True)
    target_variable = target_variable.reset_index(drop=True)

    print(f"  After split:")
    print(f"    X_training: {X_training.shape}")
    print(f"    X_testing: {X_testing.shape}")
    print(f"    target_variable: {target_variable.shape}")

    # --- Column Consistency Check ---
    print("Checking column consistency...")
    train_cols = set(X_training.columns)
    test_cols = set(X_testing.columns)

    cols_in_train_not_test = train_cols - test_cols
    cols_in_test_not_train = test_cols - train_cols
    common_cols = train_cols & test_cols

    print(f"  Columns in train but not in test: {cols_in_train_not_test}")
    print(f"  Columns in test but not in train: {cols_in_test_not_train}")
    print(f"  Common columns: {len(common_cols)}")

    # Ensure column consistency
    if cols_in_train_not_test or cols_in_test_not_train:
        print("  Fixing column inconsistencies...")

        # Add missing columns to test
        for col in cols_in_train_not_test:
            X_testing[col] = 0
            print(f"    Added missing column '{col}' to test with value 0")

        # Remove extra columns from test (shouldn't happen but just in case)
        for col in cols_in_test_not_train:
            X_testing = X_testing.drop(columns=[col])
            print(f"    Removed extra column '{col}' from test")

        # Ensure same column order
        X_testing = X_testing[X_training.columns]
        print(f"  After consistency fix:")
        print(f"    X_training: {X_training.shape}")
        print(f"    X_testing: {X_testing.shape}")

    # --- Final Data Quality Checks ---
    print("Final data quality checks...")

    # Check for missing values
    train_missing_final = X_training.isnull().sum().sum()
    test_missing_final = X_testing.isnull().sum().sum()
    target_missing_final = target_variable.isnull().sum()

    print(f"  Missing values in final datasets:")
    print(f"    X_training: {train_missing_final}")
    print(f"    X_testing: {test_missing_final}")
    print(f"    target_variable: {target_missing_final}")

    if train_missing_final > 0:
        print("  Remaining missing in X_training:")
        missing_cols = X_training.isnull().sum()
        for col, count in missing_cols[missing_cols > 0].items():
            print(f"    {col}: {count}")

    if test_missing_final > 0:
        print("  Remaining missing in X_testing:")
        missing_cols = X_testing.isnull().sum()
        for col, count in missing_cols[missing_cols > 0].items():
            print(f"    {col}: {count}")

    # Check data types
    print(f"  Final data types:")
    print(f"    X_training:")
    train_dtypes = X_training.dtypes.value_counts()
    for dtype, count in train_dtypes.items():
        print(f"      {dtype}: {count} columns")

    print(f"    X_testing:")
    test_dtypes = X_testing.dtypes.value_counts()
    for dtype, count in test_dtypes.items():
        print(f"      {dtype}: {count} columns")

    # Print final column list
    print(f"\nFinal columns ({len(X_training.columns)}):")
    for i, col in enumerate(X_training.columns, 1):
        print(f"  {i:3d}. {col}")

    print("Preprocessing pipeline completed successfully (data remains unscaled).")

    return X_training, target_variable, X_testing, test_hospital_ids, None

# Main execution block
if __name__ == "__main__":
    print("Enhanced Preprocessing Pipeline with Advanced Feature Engineering")
    print("="*80)

    print("Loading datasets...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print(f"Initial data shapes - Train: {train_df.shape}, Test: {test_df.shape}")

    # Show initial dataset info
    print(f"\nInitial dataset info:")
    print("TRAIN DATASET:")
    print(f"  Shape: {train_df.shape}")
    print(f"  Columns: {list(train_df.columns)}")
    print(f"  Data types: {train_df.dtypes.value_counts().to_dict()}")

    print("\nTEST DATASET:")
    print(f"  Shape: {test_df.shape}")
    print(f"  Columns: {list(test_df.columns)}")
    print(f"  Data types: {test_df.dtypes.value_counts().to_dict()}")

    print("\n" + "="*80)
    print("STARTING PREPROCESSING...")
    print("="*80)

    X_train, y_train, X_test, test_ids, scaler = execute_preprocessing(train_df, test_df)

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)

    print(f"Final processed shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  test_ids: {test_ids.shape}")

    # Additional summary statistics
    if y_train is not None:
        print(f"\nTarget variable (log-transformed) stats:")
        print(f"  Mean: {y_train.mean():.4f}")
        print(f"  Std: {y_train.std():.4f}")
        print(f"  Min: {y_train.min():.4f}")
        print(f"  Max: {y_train.max():.4f}")

    # Save processed data
    print(f"\nSaving processed datasets...")
    X_train.to_csv('X_train_processed_v5_enhanced.csv', index=False)
    X_test.to_csv('X_test_processed_v5_enhanced.csv', index=False)
    y_train.to_csv('y_train_processed_v5_enhanced.csv', index=False)
    test_ids.to_csv('test_ids_v5_enhanced.csv', index=False)

    print("Enhanced processed datasets saved successfully!")
    print("Files created:")
    print("- X_train_processed_v5_enhanced.csv")
    print("- X_test_processed_v5_enhanced.csv")
    print("- y_train_processed_v5_enhanced.csv")
    print("- test_ids_v5_enhanced.csv")

    print(f"\nData reduction summary:")
    print(f"  Original train samples: {len(train_df)}")
    print(f"  Final train samples: {len(X_train)}")
    if len(train_df) != len(X_train):
        reduction = len(train_df) - len(X_train)
        print(f"  Samples removed: {reduction} ({(reduction/len(train_df))*100:.2f}%)")

    print("\n" + "="*80)
    print("ENHANCED PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*80)