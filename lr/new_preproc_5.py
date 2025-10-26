import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(123)

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
    """
    print("Initiating preprocessing pipeline...")

    # --- Target Variable Processing ---
    # Remove non-positive transport costs
    training_data = training_data[training_data['Transport_Cost'] > 0].copy()
    target_variable = np.log1p(training_data['Transport_Cost'])
    training_data = training_data.drop(columns=['Transport_Cost'])

    # --- Preserve Test Identifiers ---
    test_hospital_ids = testing_data['Hospital_Id']

    # --- Missing Value Imputation Strategy ---
    numerical_missing_features = ['Supplier_Reliability', 'Equipment_Height', 'Equipment_Width', 'Equipment_Weight', 'Equipment_Value']
    categorical_missing_features = ['Transport_Method', 'Equipment_Type', 'Rural_Hospital']

    print("Computing imputation statistics from training data exclusively...")
    fill_values = {}

    # Calculate medians for numerical features
    for feature in numerical_missing_features:
        fill_values[feature] = training_data[feature].median()

    # Calculate modes for categorical features
    for feature in categorical_missing_features:
        mode_result = training_data[feature].mode()
        fill_values[feature] = mode_result[0] if not mode_result.empty else 'Missing'

    print("Applying computed fill values to both datasets...")
    for feature, fill_value in fill_values.items():
        training_data[feature] = training_data[feature].fillna(fill_value)
        testing_data[feature] = testing_data[feature].fillna(fill_value)

    # --- Feature Engineering Application ---
    print("Executing feature engineering transformations...")
    training_data = create_engineered_features(training_data)
    testing_data = create_engineered_features(testing_data)

    # --- Cyclical Encoding for Temporal Features ---
    training_data = encode_cyclical_variables(training_data, 'Order_Month_Num', 12)
    testing_data = encode_cyclical_variables(testing_data, 'Order_Month_Num', 12)
    training_data = encode_cyclical_variables(training_data, 'Order_Weekday', 7)
    testing_data = encode_cyclical_variables(testing_data, 'Order_Weekday', 7)

    # --- Dataset Combination for Consistent Encoding ---
    combined_dataset = pd.concat([training_data, testing_data], axis=0).reset_index(drop=True)

    # --- Column Removal ---
    columns_to_remove = [
        'Hospital_Id', 'Supplier_Name', 'Hospital_Location', 'Supplier_Reliability',
        'Order_Placed_Date', 'Delivery_Date'
    ]
    combined_dataset = combined_dataset.drop(columns=[col for col in columns_to_remove if col in combined_dataset.columns], errors='ignore')

    # --- Categorical Variable Encoding ---
    binary_encoding_map = {'Yes': 1, 'No': 0}
    combined_dataset['Rural_Hospital'] = combined_dataset['Rural_Hospital'].map(binary_encoding_map)

    binary_feature_list = ['CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service', 'Fragile_Equipment']
    for feature in binary_feature_list:
        combined_dataset[feature] = combined_dataset[feature].map(binary_encoding_map)

    # One-hot encoding for remaining categorical features
    categorical_features_for_ohe = ['Equipment_Type', 'Transport_Method', 'Hospital_Info', 'Order_Year_Num']

    combined_dataset = pd.get_dummies(combined_dataset, columns=categorical_features_for_ohe, drop_first=True, dtype=int)

    # --- Final Dataset Split ---
    print("Performing final dataset separation...")
    X_training = combined_dataset.iloc[:len(target_variable)].copy()
    X_testing = combined_dataset.iloc[len(target_variable):].copy()

    # Ensure column consistency between train and test
    missing_test_columns = set(X_training.columns) - set(X_testing.columns)
    for column in missing_test_columns:
        X_testing[column] = 0
    X_testing = X_testing[X_training.columns]

    print("Preprocessing pipeline completed successfully (data remains unscaled).")

    return X_training, target_variable, X_testing, test_hospital_ids, None

# Main execution block
if __name__ == "__main__":
    print("Loading datasets...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print(f"Initial data shapes - Train: {train_df.shape}, Test: {test_df.shape}")

    X_train, y_train, X_test, test_ids, scaler = execute_preprocessing(train_df, test_df)

    print(f"Final processed shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Target variable shape: {y_train.shape}")

    # Save processed data
    X_train.to_csv('X_train_processed_v5.csv', index=False)
    X_test.to_csv('X_test_processed_v5.csv', index=False)
    y_train.to_csv('y_train_processed_v5.csv', index=False)
    test_ids.to_csv('test_ids_v5.csv', index=False)

    print("Processed datasets saved successfully!")
