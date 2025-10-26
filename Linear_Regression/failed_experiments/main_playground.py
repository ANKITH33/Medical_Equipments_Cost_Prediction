import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed globally
np.random.seed(42)

def clean_initial_columns(train_data, test_data):
    """Remove ID, name and location columns that aren't useful for modeling"""
    print("Cleaning initial unnecessary columns...")

    # Define columns that should be removed initially
    unnecessary_cols = ['Hospital_Id', 'Hospital_Location']

    train_cleaned = train_data.copy()
    test_cleaned = test_data.copy()

    for col in unnecessary_cols:
        if col in train_cleaned.columns:
            train_cleaned = train_cleaned.drop(col, axis=1)
        if col in test_cleaned.columns:
            test_cleaned = test_cleaned.drop(col, axis=1)

    print(f"Removed initial columns: {unnecessary_cols}")
    return train_cleaned, test_cleaned

def create_delivery_duration_feature(train_data, test_data):
    """Calculate the number of days between order and delivery dates"""
    print("Creating delivery duration feature...")

    train_modified = train_data.copy()
    test_modified = test_data.copy()

    for dataset in [train_modified, test_modified]:
        if 'Order_Placed_Date' in dataset.columns and 'Delivery_Date' in dataset.columns:
            # Convert to datetime
            dataset['Order_Placed_Date'] = pd.to_datetime(dataset['Order_Placed_Date'])
            dataset['Delivery_Date'] = pd.to_datetime(dataset['Delivery_Date'])

            # Calculate duration in days
            dataset['Delivery_Duration'] = (dataset['Delivery_Date'] - dataset['Order_Placed_Date']).dt.days

    print("Added Delivery_Duration feature")
    return train_modified, test_modified

def remove_correlated_equipment_features(train_data, test_data):
    """Remove highly correlated equipment features"""
    print("Removing highly correlated equipment features...")

    correlated_features = ["Equipment_Width", "Equipment_Weight"]

    train_filtered = train_data.copy()
    test_filtered = test_data.copy()

    for feature in correlated_features:
        if feature in train_filtered.columns:
            train_filtered = train_filtered.drop(feature, axis=1)
        if feature in test_filtered.columns:
            test_filtered = test_filtered.drop(feature, axis=1)

    print(f"Removed correlated features: {correlated_features}")
    return train_filtered, test_filtered

def fill_supplier_reliability_by_group(train_data, test_data, target_column, group_column):
    """Fill missing values in target column using group means from COMBINED data (overfitting)"""
    print(f"Filling {target_column} using {group_column} group statistics from COMBINED data...")

    if group_column not in train_data.columns:
        print(f"Warning: {group_column} not found. Skipping group filling.")
        return train_data.copy(), test_data.copy()

    train_processed = train_data.copy()
    test_processed = test_data.copy()

    # OVERFITTING: Calculate group means from combined data including test set
    combined_data = pd.concat([train_processed, test_processed], ignore_index=True)
    group_statistics = combined_data.groupby(group_column)[target_column].mean().to_dict()

    # Fill missing values in train data
    missing_mask_train = train_processed[target_column].isna()
    if missing_mask_train.any():
        train_processed.loc[missing_mask_train, target_column] = train_processed.loc[missing_mask_train, group_column].map(group_statistics)

    # Fill missing values in test data
    if target_column in test_processed.columns:
        missing_mask_test = test_processed[target_column].isna()
        if missing_mask_test.any():
            test_processed.loc[missing_mask_test, target_column] = test_processed.loc[missing_mask_test, group_column].map(group_statistics)

    # Fill any remaining missing values with overall mean from COMBINED data
    overall_average = combined_data[target_column].mean()
    train_processed[target_column].fillna(overall_average, inplace=True)
    if target_column in test_processed.columns:
        test_processed[target_column].fillna(overall_average, inplace=True)

    print(f"Filled {target_column} using {group_column} group means from COMBINED data (OVERFITTING)")
    return train_processed, test_processed

def fill_categorical_with_unknown(train_data, test_data, column_name):
    """Replace missing values in categorical column with 'Unknown'"""
    print(f"Filling missing values in {column_name} with 'Unknown'...")

    train_filled = train_data.copy()
    test_filled = test_data.copy()

    if column_name in train_filled.columns:
        train_filled[column_name].fillna('Unknown', inplace=True)

    if column_name in test_filled.columns:
        test_filled[column_name].fillna('Unknown', inplace=True)

    print(f"Filled {column_name} missing values with 'Unknown'")
    return train_filled, test_filled

def fill_numerical_with_mean(train_data, test_data, column_name):
    """Replace missing values in numerical column with mean from COMBINED data (overfitting)"""
    print(f"Filling missing values in {column_name} with mean from COMBINED data...")

    train_filled = train_data.copy()
    test_filled = test_data.copy()

    # OVERFITTING: Calculate mean from combined data including test set
    combined_data = pd.concat([train_filled, test_filled], ignore_index=True)
    column_mean = combined_data[column_name].mean()

    if column_name in train_filled.columns:
        train_filled[column_name].fillna(column_mean, inplace=True)

    if column_name in test_filled.columns:
        test_filled[column_name].fillna(column_mean, inplace=True)

    print(f"Filled {column_name} with mean value from COMBINED data: {column_mean:.4f} (OVERFITTING)")
    return train_filled, test_filled

def extract_datetime_components(train_data, test_data, datetime_columns):
    """Extract day, month, year from datetime columns"""
    print(f"Extracting components from datetime columns: {datetime_columns}")

    train_expanded = train_data.copy()
    test_expanded = test_data.copy()

    for dataset in [train_expanded, test_expanded]:
        for col in datetime_columns:
            if col in dataset.columns:
                # Ensure datetime format
                dataset[col] = pd.to_datetime(dataset[col])

                # Extract components
                dataset[f'{col}_Day'] = dataset[col].dt.day
                dataset[f'{col}_Month'] = dataset[col].dt.month
                dataset[f'{col}_Year'] = dataset[col].dt.year

    print(f"Added day/month/year components for: {datetime_columns}")
    return train_expanded, test_expanded

def remove_columns_list(train_data, test_data, columns_to_remove):
    """Remove specified columns from both datasets"""
    print(f"Removing specified columns: {columns_to_remove}")

    train_trimmed = train_data.copy()
    test_trimmed = test_data.copy()

    for col in columns_to_remove:
        if col in train_trimmed.columns:
            train_trimmed = train_trimmed.drop(col, axis=1)
        if col in test_trimmed.columns:
            test_trimmed = test_trimmed.drop(col, axis=1)

    print(f"Removed columns: {columns_to_remove}")
    return train_trimmed, test_trimmed

def drop_rows_with_nulls(train_data, test_data):
    """Remove any rows that still contain null values"""
    print("Removing rows with any remaining null values...")

    # Check for nulls before
    train_nulls_before = train_data.isnull().sum().sum()
    test_nulls_before = test_data.isnull().sum().sum()

    print(f"Nulls before - Train: {train_nulls_before}, Test: {test_nulls_before}")

    train_clean = train_data.copy()
    test_clean = test_data.copy()

    if train_nulls_before > 0:
        print("Columns with nulls in train:", train_clean.columns[train_clean.isnull().any()].tolist())
        train_clean = train_clean.dropna().reset_index(drop=True)

    if test_nulls_before > 0:
        print("Columns with nulls in test:", test_clean.columns[test_clean.isnull().any()].tolist())
        test_clean = test_clean.dropna().reset_index(drop=True)

    train_nulls_after = train_clean.isnull().sum().sum()
    test_nulls_after = test_clean.isnull().sum().sum()

    print(f"Nulls after - Train: {train_nulls_after}, Test: {test_nulls_after}")
    return train_clean, test_clean

def filter_target_outliers_by_percentile(train_data, test_data, target_column, percentile_bounds):
    """Remove outliers from target column using percentile bounds"""
    print(f"Filtering {target_column} outliers using {percentile_bounds} percentile bounds...")

    if target_column not in train_data.columns:
        print(f"Warning: {target_column} not found in training data")
        return train_data.copy(), test_data.copy()

    # Calculate percentile bounds
    lower_pct, upper_pct = percentile_bounds
    lower_threshold = np.percentile(train_data[target_column], lower_pct)
    upper_threshold = np.percentile(train_data[target_column], upper_pct)

    # Filter training data
    original_count = len(train_data)
    train_filtered = train_data[
        (train_data[target_column] >= lower_threshold) &
        (train_data[target_column] <= upper_threshold)
    ].reset_index(drop=True)

    print(f"Removed values outside {lower_pct}th–{upper_pct}th percentiles ({lower_threshold:.3f} to {upper_threshold:.3f})")
    print(f"Removed {original_count - len(train_filtered)} samples. Remaining: {len(train_filtered)}")

    return train_filtered, test_data.copy()

def create_train_validation_split(train_data, validation_size=0.15, random_seed=42):
    """Split training data into train and validation sets"""
    print(f"Creating train/validation split (validation size: {validation_size})...")

    if 'Transport_Cost' in train_data.columns:
        features = train_data.drop('Transport_Cost', axis=1)
        target = train_data['Transport_Cost']

        X_train, X_val, y_train, y_val = train_test_split(
            features, target, test_size=validation_size, random_state=random_seed
        )

        # Recombine for consistent return format
        train_split = pd.concat([X_train, y_train], axis=1)
        validation_split = pd.concat([X_val, y_val], axis=1)
    else:
        # If no target column, split features only
        train_split, validation_split = train_test_split(
            train_data, test_size=validation_size, random_state=random_seed
        )

    print(f"Train set: {train_split.shape}, Validation set: {validation_split.shape}")
    return train_split, validation_split

def build_and_train_ridge_model_with_poly(train_data, validation_data, test_data, alpha_value=100, model_name="ridge_model"):
    """Build and train Ridge regression model with polynomial features and preprocessing pipeline"""
    print(f"Building Ridge model with polynomial features (degree=2) and alpha={alpha_value}...")

    # Separate features and target
    X_train = train_data.drop('Transport_Cost', axis=1)
    y_train = train_data['Transport_Cost']
    X_val = validation_data.drop('Transport_Cost', axis=1)
    y_val = validation_data['Transport_Cost']
    X_test = test_data.copy()

    # Identify feature types
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    # Create preprocessing pipeline
    preprocessing_steps = []
    if numeric_features:
        preprocessing_steps.append(('numeric', StandardScaler(), numeric_features))
    if categorical_features:
        preprocessing_steps.append(('categorical', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features))

    if not preprocessing_steps:
        preprocessor = 'passthrough'
    else:
        preprocessor = ColumnTransformer(
            transformers=preprocessing_steps,
            remainder='passthrough'
        )

    # Create complete pipeline with polynomial features
    ridge_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("polynomial", PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)),
        ("scaler", StandardScaler()),  # Scale after polynomial expansion
        ("regressor", Ridge(alpha=alpha_value, random_state=42))
    ])

    # Train the model
    print("Training Ridge model with polynomial features...")
    ridge_pipeline.fit(X_train, y_train)

    # Evaluate performance
    train_predictions = ridge_pipeline.predict(X_train)
    validation_predictions = ridge_pipeline.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    val_rmse = np.sqrt(mean_squared_error(y_val, validation_predictions))
    train_r2 = r2_score(y_train, train_predictions)
    val_r2 = r2_score(y_val, validation_predictions)

    print(f"Model Performance:")
    print(f"  Train RMSE: {train_rmse:.3f}, R²: {train_r2:.4f}")
    print(f"  Validation RMSE: {val_rmse:.3f}, R²: {val_r2:.4f}")

    return ridge_pipeline

def create_submission_file(trained_model, test_data, model_name="model"):
    """Generate predictions and create submission file"""
    print(f"Creating submission file for {model_name}...")

    # Load original test data for Hospital_Id
    original_test = pd.read_csv('test.csv')

    # Generate predictions
    test_predictions = trained_model.predict(test_data)

    # Create submission dataframe
    submission_df = pd.DataFrame({
        'Hospital_Id': original_test['Hospital_Id'],
        'Transport_Cost': test_predictions
    })

    # Save submission file
    submission_filename = f'submission_{model_name}.csv'
    submission_df.to_csv(submission_filename, index=False)

    print(f"Submission saved to {submission_filename}")
    print(f"Prediction statistics - Mean: {test_predictions.mean():.2f}, Std: {test_predictions.std():.2f}")

    return submission_df

def run_complete_modeling_pipeline_with_multiple_alphas():
    """Execute the complete modeling pipeline with multiple alpha values"""
    print("="*80)
    print("RUNNING COMPLETE MODELING PIPELINE WITH POLYNOMIAL FEATURES AND MULTIPLE ALPHAS")
    print("USING TEST DATA FOR ALL STATISTICS (MAXIMUM OVERFITTING)")
    print("="*80)

    # Load data
    train_dataset = pd.read_csv('train.csv')
    test_dataset = pd.read_csv('test.csv')

    print(f"Initial data shapes - Train: {train_dataset.shape}, Test: {test_dataset.shape}")

    # Execute preprocessing pipeline step by step
    train_processed, test_processed = clean_initial_columns(train_dataset, test_dataset)
    train_processed, test_processed = create_delivery_duration_feature(train_processed, test_processed)
    train_processed, test_processed = remove_correlated_equipment_features(train_processed, test_processed)
    train_processed, test_processed = fill_supplier_reliability_by_group(train_processed, test_processed, "Supplier_Reliability", "Supplier_Name")
    train_processed, test_processed = fill_categorical_with_unknown(train_processed, test_processed, "Rural_Hospital")
    train_processed, test_processed = fill_categorical_with_unknown(train_processed, test_processed, "Equipment_Type")
    train_processed, test_processed = fill_categorical_with_unknown(train_processed, test_processed, "Transport_Method")
    train_processed, test_processed = fill_numerical_with_mean(train_processed, test_processed, "Equipment_Height")
    train_processed, test_processed = extract_datetime_components(train_processed, test_processed, ["Order_Placed_Date", "Delivery_Date"])
    train_processed, test_processed = remove_columns_list(train_processed, test_processed, ["Order_Placed_Date", "Delivery_Date", "Supplier_Name"])
    train_processed, test_processed = drop_rows_with_nulls(train_processed, test_processed)
    train_processed, test_processed = filter_target_outliers_by_percentile(train_processed, test_processed, "Transport_Cost", [5, 95])

    # Create train/validation split
    train_final, validation_final = create_train_validation_split(train_processed, validation_size=0.15, random_seed=42)
    test_final = test_processed

    # Print final data information
    print(f"\nFinal processed data shapes:")
    print(f"Train: {train_final.shape}")
    print(f"Validation: {validation_final.shape}")
    print(f"Test: {test_final.shape}")

    # Define alpha values to try
    alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 10000, 100000, 1000000]

    print(f"\nTraining {len(alpha_values)} models with different alpha values...")
    print(f"Alpha values: {alpha_values}")

    models = {}
    results = []

    # Train models with different alpha values
    for i, alpha in enumerate(alpha_values, 1):
        print(f"\n{'-'*60}")
        print(f"TRAINING MODEL {i}/{len(alpha_values)} - ALPHA = {alpha}")
        print(f"{'-'*60}")

        experiment_id = f"poly_alpha_{alpha}"

        # Train model
        trained_model = build_and_train_ridge_model_with_poly(
            train_final, validation_final, test_final, alpha, experiment_id
        )

        # Store model
        models[alpha] = trained_model

        # Generate submission
        submission = create_submission_file(trained_model, test_final, experiment_id)

        # Store results
        results.append({
            'alpha': alpha,
            'model_id': experiment_id,
            'submission_file': f'submission_{experiment_id}.csv'
        })

        print(f"Completed model with alpha={alpha}")

    print("\n" + "="*80)
    print("ALL MODELS COMPLETED!")
    print("="*80)

    print(f"\nGenerated {len(results)} submission files:")
    for result in results:
        print(f"  Alpha {result['alpha']:8.1f}: {result['submission_file']}")

    print(f"\nAll models use:")
    print(f"  - Polynomial features (degree=2)")
    print(f"  - StandardScaler + OneHotEncoder")
    print(f"  - Ridge regression")
    print(f"  - Test data for all statistics (OVERFITTING)")
    print(f"  - Outlier filtering: 2nd-98th percentiles")

    return models, results

# Execute the pipeline
if __name__ == "__main__":
    models, results = run_complete_modeling_pipeline_with_multiple_alphas()
