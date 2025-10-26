import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(123)

def analyze_raw_data_distribution(train_df, test_df):
    """
    Analyze raw data before preprocessing - combined from eda_irr.py
    """
    print("="*80)
    print("RAW DATA EXPLORATORY ANALYSIS")
    print("="*80)

    print(f"Train dataset shape: {train_df.shape}")
    print(f"Test dataset shape: {test_df.shape}")

    # Basic info about train dataset
    print("\nTRAIN DATASET INFO:")
    print("="*50)
    print(train_df.info())

    # Check for missing values
    print("\nTRAIN DATASET MISSING VALUES:")
    print("="*50)
    missing_train = train_df.isnull().sum()
    missing_percent_train = (missing_train / len(train_df)) * 100
    missing_df_train = pd.DataFrame({
        'Column': missing_train.index,
        'Missing_Count': missing_train.values,
        'Missing_Percentage': missing_percent_train.values
    }).sort_values('Missing_Count', ascending=False)
    print(missing_df_train)

    # Check unique values for categorical columns
    print("\nTRAIN DATASET UNIQUE VALUES (for columns with ≤ 20 unique values):")
    print("="*70)
    for col in train_df.columns:
        unique_count = train_df[col].nunique()
        if unique_count <= 20:
            print(f"\n{col} ({unique_count} unique values):")
            print(train_df[col].value_counts().head(20))

    # Compare columns between train and test
    print("\nCOLUMN COMPARISON BETWEEN TRAIN AND TEST:")
    print("="*50)
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    print(f"Columns in train but not in test: {train_cols - test_cols}")
    print(f"Columns in test but not in train: {test_cols - train_cols}")
    print(f"Common columns: {len(train_cols & test_cols)}")

    # Check target variable if it exists in train
    if 'Transport_Cost' in train_df.columns:
        print("\nTARGET VARIABLE (Transport_Cost) ANALYSIS:")
        print("="*50)
        print(f"Mean: {train_df['Transport_Cost'].mean():.2f}")
        print(f"Median: {train_df['Transport_Cost'].median():.2f}")
        print(f"Std: {train_df['Transport_Cost'].std():.2f}")
        print(f"Min: {train_df['Transport_Cost'].min():.2f}")
        print(f"Max: {train_df['Transport_Cost'].max():.2f}")
        print(f"Skewness: {train_df['Transport_Cost'].skew():.2f}")
        print(f"Kurtosis: {train_df['Transport_Cost'].kurtosis():.2f}")

        # Percentiles for outlier detection
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print("\nPercentiles:")
        for p in percentiles:
            val = np.percentile(train_df['Transport_Cost'].dropna(), p)
            print(f"{p:2d}th percentile: {val:.2f}")

def analyze_transport_cost_distribution(y_train):
    """
    Detailed transport cost analysis - from tc_analysis_1.py
    """
    print("\n" + "="*80)
    print("TRANSPORT COST DISTRIBUTION ANALYSIS")
    print("="*80)

    print(f"Count: {len(y_train)}")
    print(f"Mean: {y_train.mean():,.2f}")
    print(f"Median: {y_train.median():,.2f}")
    print(f"Std: {y_train.std():,.2f}")
    print(f"Min: {y_train.min():,.2f}")
    print(f"Max: {y_train.max():,.2f}")
    print(f"Skewness: {y_train.skew():.3f}")
    print(f"Kurtosis: {y_train.kurtosis():.3f}")

    # Check for negative and zero values
    negative_count = (y_train < 0).sum()
    zero_count = (y_train == 0).sum()
    print(f"\nNegative values: {negative_count} ({negative_count/len(y_train)*100:.2f}%)")
    print(f"Zero values: {zero_count} ({zero_count/len(y_train)*100:.2f}%)")

    # Outlier analysis using IQR
    Q1 = y_train.quantile(0.25)
    Q3 = y_train.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_iqr = ((y_train < lower_bound) | (y_train > upper_bound)).sum()
    print(f"\nOutliers (IQR method): {outliers_iqr} ({outliers_iqr/len(y_train)*100:.2f}%)")
    print(f"IQR bounds: [{lower_bound:,.2f}, {upper_bound:,.2f}]")

    # Analysis for different percentile thresholds
    print(f"\nOUTLIER ANALYSIS FOR DIFFERENT PERCENTILES:")
    print("="*60)
    thresholds = [(2.5, 97.5), (5, 95), (10, 90), (1, 99)]

    for lower_p, upper_p in thresholds:
        lower_threshold = np.percentile(y_train, lower_p)
        upper_threshold = np.percentile(y_train, upper_p)

        outliers = ((y_train < lower_threshold) | (y_train > upper_threshold)).sum()
        remaining = len(y_train) - outliers

        print(f"\nRemoving {lower_p}th and {upper_p}th percentile outliers:")
        print(f"  Thresholds: [{lower_threshold:,.2f}, {upper_threshold:,.2f}]")
        print(f"  Outliers removed: {outliers} ({outliers/len(y_train)*100:.1f}%)")
        print(f"  Remaining samples: {remaining} ({remaining/len(y_train)*100:.1f}%)")

def encode_cyclical_variables(dataframe, column_name, maximum_value):
    """
    Create sine and cosine encodings for cyclical variables.
    """
    dataframe[column_name + '_sine'] = np.sin(2 * np.pi * dataframe[column_name] / maximum_value)
    dataframe[column_name + '_cosine'] = np.cos(2 * np.pi * dataframe[column_name] / maximum_value)
    return dataframe

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
        print(f"  Correcting {invalid_dates_mask.sum()} rows with invalid date sequences...")
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

def simple_preproc_approach(X_train, X_test):
    """
    Alternative simple preprocessing approach from preproc_0.py
    No feature engineering, just basic encoding with dummy_na
    """
    print("\n" + "="*50)
    print("ALTERNATIVE: SIMPLE PREPROCESSING APPROACH")
    print("="*50)

    # Identify categorical columns (object dtype)
    categorical_cols_train = X_train.select_dtypes(include=['object']).columns.tolist()
    categorical_cols_test = X_test.select_dtypes(include=['object']).columns.tolist()

    print(f"Categorical columns in train: {categorical_cols_train}")
    print(f"Categorical columns in test: {categorical_cols_test}")

    # Get all categorical columns from both datasets
    all_categorical_cols = list(set(categorical_cols_train + categorical_cols_test))
    print(f"All categorical columns to encode: {all_categorical_cols}")

    # Add a flag to identify train vs test rows (preproc_0 approach)
    X_train_simple = X_train.copy()
    X_test_simple = X_test.copy()

    X_train_simple['is_train'] = 1
    X_test_simple['is_train'] = 0

    # Combine datasets
    combined_df = pd.concat([X_train_simple, X_test_simple], ignore_index=True, sort=False)
    print(f"Combined dataset shape: {combined_df.shape}")

    # Get categorical columns that exist in combined dataset
    categorical_cols = [col for col in all_categorical_cols if col in combined_df.columns]

    if categorical_cols:
        print(f"Simple encoding columns: {categorical_cols}")

        # One-hot encode with preproc_0 settings: drop_first=False, dummy_na=True
        combined_encoded = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=False, dummy_na=True)
        print(f"Shape after simple one-hot encoding: {combined_encoded.shape}")

        # Split back into train and test
        train_mask = combined_encoded['is_train'] == 1
        test_mask = combined_encoded['is_train'] == 0

        X_train_simple_encoded = combined_encoded[train_mask].drop('is_train', axis=1)
        X_test_simple_encoded = combined_encoded[test_mask].drop('is_train', axis=1)
    else:
        print("No categorical columns to encode in simple approach")
        X_train_simple_encoded = X_train_simple.drop('is_train', axis=1)
        X_test_simple_encoded = X_test_simple.drop('is_train', axis=1)

    print(f"Simple approach results - Train: {X_train_simple_encoded.shape}, Test: {X_test_simple_encoded.shape}")

    return X_train_simple_encoded, X_test_simple_encoded

def check_nan_values_comprehensive(X_train, X_test, y_train):
    """
    Comprehensive NaN checking - from check_nan_4.py
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE NaN VALUE ANALYSIS")
    print("="*80)

    print(f"Dataset shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")

    # Check X_train for NaN values
    print("\nCHECKING X_TRAIN FOR NaN VALUES:")
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

    # Check X_test for NaN values
    print("\nCHECKING X_TEST FOR NaN VALUES:")
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

    # Check y_train for NaN values
    print("\nCHECKING Y_TRAIN FOR NaN VALUES:")
    print("-" * 40)

    if isinstance(y_train, pd.DataFrame):
        y_train_nan_count = y_train.isnull().sum().sum()
    else:
        y_train_nan_count = y_train.isnull().sum()

    print(f"Total NaN values in y_train: {y_train_nan_count}")

    if y_train_nan_count > 0:
        print("⚠ WARNING: NaN values found in y_train!")
    else:
        print("✓ No NaN values found in y_train")

    # Summary
    total_nan_all = X_train_total_nan + X_test_total_nan + y_train_nan_count

    print(f"\nSUMMARY:")
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

def impute_remaining_numerical_features(X_train, X_test):
    """
    Additional numerical imputation if needed - from impute_3.py logic
    """
    # Identify numerical columns that still have missing values
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    missing_numerical = []
    for col in numerical_cols:
        if X_train[col].isnull().sum() > 0 or X_test[col].isnull().sum() > 0:
            missing_numerical.append(col)

    if missing_numerical:
        print(f"\nApplying additional numerical imputation for: {missing_numerical}")

        # Calculate median values from train set
        imputation_values = {}
        for col in missing_numerical:
            median_val = X_train[col].median()
            imputation_values[col] = median_val
            print(f"  {col}: median = {median_val:.4f}")

        # Apply imputation
        for col, median_val in imputation_values.items():
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

        print("Additional numerical imputation completed.")

        # Show some statistics of imputed columns
        print(f"\nStatistics of imputed numerical columns (train set):")
        for col in imputation_values.keys():
            print(f"\n{col}:")
            print(f"  Mean: {X_train[col].mean():.4f}")
            print(f"  Median: {X_train[col].median():.4f}")
            print(f"  Std: {X_train[col].std():.4f}")
            print(f"  Min: {X_train[col].min():.4f}")
            print(f"  Max: {X_train[col].max():.4f}")

    return X_train, X_test

def remove_outliers_percentile_based(X_train, y_train, lower_percentile=2.5, upper_percentile=97.5):
    """
    Remove outliers based on percentiles - from remove_outliers_2.py
    """
    print(f"\n" + "="*50)
    print("OUTLIER REMOVAL BASED ON PERCENTILES")
    print("="*50)

    print(f"Original shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    # Calculate percentiles
    lower_threshold = np.percentile(y_train, lower_percentile)
    upper_threshold = np.percentile(y_train, upper_percentile)

    print(f"\nTarget variable statistics before outlier removal:")
    print(f"Mean: {y_train.mean():.2f}")
    print(f"Std: {y_train.std():.2f}")
    print(f"Min: {y_train.min():.2f}")
    print(f"Max: {y_train.max():.2f}")

    print(f"\nPercentile thresholds:")
    print(f"{lower_percentile}th percentile (lower threshold): {lower_threshold:.2f}")
    print(f"{upper_percentile}th percentile (upper threshold): {upper_threshold:.2f}")

    # Identify outliers
    outlier_mask_lower = y_train <= lower_threshold
    outlier_mask_upper = y_train >= upper_threshold
    outlier_mask = outlier_mask_lower | outlier_mask_upper

    # Count outliers
    num_lower_outliers = outlier_mask_lower.sum()
    num_upper_outliers = outlier_mask_upper.sum()
    total_outliers = outlier_mask.sum()

    print(f"\nOutlier analysis:")
    print(f"Lower outliers (≤ {lower_threshold:.2f}): {num_lower_outliers}")
    print(f"Upper outliers (≥ {upper_threshold:.2f}): {num_upper_outliers}")
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

    return X_train_filtered, y_train_filtered

def execute_preprocessing(training_data, testing_data, use_advanced=True, use_outlier_removal=False):
    """
    Complete preprocessing pipeline with comprehensive analysis and feature engineering.
    Now includes all approaches merged together.
    """
    print("="*80)
    print("COMPREHENSIVE MERGED PREPROCESSING PIPELINE")
    print("="*80)

    # Step 1: Raw data analysis
    analyze_raw_data_distribution(training_data, testing_data)

    # Step 2: Target variable processing and analysis
    if 'Transport_Cost' in training_data.columns:
        # Analyze raw target distribution
        analyze_transport_cost_distribution(training_data['Transport_Cost'])

        # Store original target for simple approach
        original_target = training_data['Transport_Cost'].copy()

        if use_advanced:
            # Remove non-positive transport costs
            print(f"\nRemoving non-positive transport costs...")
            original_count = len(training_data)
            training_data = training_data[training_data['Transport_Cost'] > 0].copy()
            removed_count = original_count - len(training_data)
            print(f"Removed {removed_count} rows with non-positive transport costs")

            # Apply log transformation to target
            target_variable = np.log1p(training_data['Transport_Cost'])
            training_data = training_data.drop(columns=['Transport_Cost'])

            # Analyze transformed target
            print(f"\nTransformed target statistics (log1p):")
            print(f"Mean: {target_variable.mean():.4f}")
            print(f"Std: {target_variable.std():.4f}")
            print(f"Min: {target_variable.min():.4f}")
            print(f"Max: {target_variable.max():.4f}")
        else:
            # Simple approach - just separate target
            target_variable = training_data['Transport_Cost'].copy()
            training_data = training_data.drop(columns=['Transport_Cost'])

    # Step 3: Preserve test identifiers
    test_hospital_ids = testing_data['Hospital_Id']

    if use_advanced:
        # Step 4: Advanced missing value imputation strategy
        print(f"\n" + "="*50)
        print("ADVANCED MISSING VALUE IMPUTATION")
        print("="*50)

        numerical_missing_features = ['Supplier_Reliability', 'Equipment_Height', 'Equipment_Width', 'Equipment_Weight', 'Equipment_Value']
        categorical_missing_features = ['Transport_Method', 'Equipment_Type', 'Rural_Hospital']

        print("Computing imputation statistics from training data exclusively...")
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
                print(f"  {feature}: mode = {fill_values[feature]}")

        print("Applying computed fill values to both datasets...")
        for feature, fill_value in fill_values.items():
            if feature in training_data.columns:
                training_data[feature] = training_data[feature].fillna(fill_value)
            if feature in testing_data.columns:
                testing_data[feature] = testing_data[feature].fillna(fill_value)

        # Step 5: Advanced feature engineering
        print(f"\n" + "="*50)
        print("ADVANCED FEATURE ENGINEERING")
        print("="*50)
        print("Executing feature engineering transformations...")
        training_data = create_engineered_features(training_data)
        testing_data = create_engineered_features(testing_data)

        # Step 6: Cyclical encoding for temporal features
        print("Applying cyclical encoding for temporal features...")
        training_data = encode_cyclical_variables(training_data, 'Order_Month_Num', 12)
        testing_data = encode_cyclical_variables(testing_data, 'Order_Month_Num', 12)
        training_data = encode_cyclical_variables(training_data, 'Order_Weekday', 7)
        testing_data = encode_cyclical_variables(testing_data, 'Order_Weekday', 7)

        # Step 7: Dataset combination for consistent encoding
        print("Combining datasets for consistent categorical encoding...")
        combined_dataset = pd.concat([training_data, testing_data], axis=0).reset_index(drop=True)

        # Step 8: Column removal
        columns_to_remove = [
            'Hospital_Id', 'Supplier_Name', 'Hospital_Location', 'Supplier_Reliability',
            'Order_Placed_Date', 'Delivery_Date'
        ]
        combined_dataset = combined_dataset.drop(columns=[col for col in columns_to_remove if col in combined_dataset.columns], errors='ignore')

        # Step 9: Advanced categorical variable encoding
        print("Advanced encoding categorical variables...")
        binary_encoding_map = {'Yes': 1, 'No': 0}
        combined_dataset['Rural_Hospital'] = combined_dataset['Rural_Hospital'].map(binary_encoding_map)

        binary_feature_list = ['CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service', 'Fragile_Equipment']
        for feature in binary_feature_list:
            if feature in combined_dataset.columns:
                combined_dataset[feature] = combined_dataset[feature].map(binary_encoding_map)

        # One-hot encoding for remaining categorical features
        categorical_features_for_ohe = ['Equipment_Type', 'Transport_Method', 'Hospital_Info', 'Order_Year_Num']
        categorical_features_for_ohe = [col for col in categorical_features_for_ohe if col in combined_dataset.columns]

        if categorical_features_for_ohe:
            print(f"Advanced one-hot encoding: {categorical_features_for_ohe}")
            combined_dataset = pd.get_dummies(combined_dataset, columns=categorical_features_for_ohe, drop_first=True, dtype=int)

        # Step 10: Final dataset split
        print("Performing final dataset separation...")
        X_training = combined_dataset.iloc[:len(target_variable)].copy()
        X_testing = combined_dataset.iloc[len(target_variable):].copy()

        # Ensure column consistency between train and test
        missing_test_columns = set(X_training.columns) - set(X_testing.columns)
        for column in missing_test_columns:
            X_testing[column] = 0
        X_testing = X_testing[X_training.columns]

    else:
        # Simple approach from preproc_0.py
        print(f"\n" + "="*50)
        print("SIMPLE PREPROCESSING APPROACH (preproc_0 style)")
        print("="*50)

        # Simple column dropping
        cols_to_drop = ['Hospital_Id', 'Supplier_Name', 'Hospital_Location', 'Order_Placed_Date', 'Delivery_Date']
        print(f"Simple dropping columns: {cols_to_drop}")

        # Drop columns from train
        training_data = training_data.drop(columns=[col for col in cols_to_drop if col in training_data.columns])
        # Drop columns from test
        testing_data = testing_data.drop(columns=[col for col in cols_to_drop if col in testing_data.columns])

        # Use simple preprocessing approach
        X_training, X_testing = simple_preproc_approach(training_data, testing_data)

    # Step 11: Additional numerical imputation if needed
    X_training, X_testing = impute_remaining_numerical_features(X_training, X_testing)

    # Step 12: Optional outlier removal
    if use_outlier_removal:
        X_training, target_variable = remove_outliers_percentile_based(X_training, target_variable)

    # Step 13: Comprehensive NaN checking
    check_nan_values_comprehensive(X_training, X_testing, target_variable)

    print(f"\n" + "="*80)
    print("MERGED PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Final shapes - X_train: {X_training.shape}, X_test: {X_testing.shape}")
    print(f"Target variable shape: {target_variable.shape}")
    print("Data remains unscaled for flexibility in modeling.")

    return X_training, target_variable, X_testing, test_hospital_ids, None

# Main execution block
if __name__ == "__main__":
    print("Loading datasets...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print(f"Initial data shapes - Train: {train_df.shape}, Test: {test_df.shape}")

    # You can choose different approaches:
    # Advanced approach (default)
    X_train, y_train, X_test, test_ids, scaler = execute_preprocessing(
        train_df.copy(), test_df.copy(),
        use_advanced=True,
        use_outlier_removal=False
    )

    # Save advanced processed data
    print(f"\nSaving advanced processed datasets...")
    X_train.to_csv('X_train_processed_comprehensive_advanced.csv', index=False)
    X_test.to_csv('X_test_processed_comprehensive_advanced.csv', index=False)
    y_train.to_csv('y_train_processed_comprehensive_advanced.csv', index=False)
    test_ids.to_csv('test_ids_comprehensive_advanced.csv', index=False)

    # Simple approach (preproc_0 style)
    print(f"\n" + "="*60)
    print("NOW RUNNING SIMPLE APPROACH")
    print("="*60)
    X_train_simple, y_train_simple, X_test_simple, test_ids_simple, scaler_simple = execute_preprocessing(
        train_df.copy(), test_df.copy(),
        use_advanced=False,
        use_outlier_removal=False
    )

    # Save simple processed data
    print(f"\nSaving simple processed datasets...")
    X_train_simple.to_csv('X_train_processed_comprehensive_simple.csv', index=False)
    X_test_simple.to_csv('X_test_processed_comprehensive_simple.csv', index=False)
    y_train_simple.to_csv('y_train_processed_comprehensive_simple.csv', index=False)
    test_ids_simple.to_csv('test_ids_comprehensive_simple.csv', index=False)

    print("\nFiles saved:")
    print("ADVANCED APPROACH:")
    print("- X_train_processed_comprehensive_advanced.csv")
    print("- X_test_processed_comprehensive_advanced.csv")
    print("- y_train_processed_comprehensive_advanced.csv")
    print("- test_ids_comprehensive_advanced.csv")
    print("\nSIMPLE APPROACH:")
    print("- X_train_processed_comprehensive_simple.csv")
    print("- X_test_processed_comprehensive_simple.csv")
    print("- y_train_processed_comprehensive_simple.csv")
    print("- test_ids_comprehensive_simple.csv")

    print("\nMerged comprehensive preprocessing completed successfully!")
    print("You now have BOTH advanced and simple preprocessing outputs!")
