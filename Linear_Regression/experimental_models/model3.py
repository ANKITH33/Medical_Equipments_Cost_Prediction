import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

def load_and_initial_cleanup(train_path='train.csv', test_path='test.csv'):
    """Load datasets and perform initial cleanup"""
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Original shapes - Train: {train_df.shape}, Test: {test_df.shape}")
    return train_df, test_df

def engineer_date_features(train_df, test_df):
    """Extract features from date columns before removal"""
    print("Engineering date features...")

    for df in [train_df, test_df]:
        if 'Order_Placed_Date' in df.columns and 'Delivery_Date' in df.columns:
            # Convert to datetime
            df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'])
            df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'])

            # Calculate delivery duration in days
            df['Delivery_Duration_Days'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days

            # Extract month and weekday features
            df['Order_Month'] = df['Order_Placed_Date'].dt.month
            df['Order_Weekday'] = df['Order_Placed_Date'].dt.dayofweek
            df['Delivery_Month'] = df['Delivery_Date'].dt.month
            df['Delivery_Weekday'] = df['Delivery_Date'].dt.dayofweek

    print("Created features: Delivery_Duration_Days, Order_Month, Order_Weekday, Delivery_Month, Delivery_Weekday")
    return train_df, test_df

def create_supplier_aggregations(train_df, test_df):
    """Create supplier-based aggregated features"""
    print("Creating supplier aggregations...")

    # Calculate supplier reliability mean grouped by supplier
    supplier_reliability_map = train_df.groupby('Supplier_Name')['Supplier_Reliability'].mean().to_dict()

    # Apply to both datasets
    train_df['Supplier_Reliability_Mean'] = train_df['Supplier_Name'].map(supplier_reliability_map)
    test_df['Supplier_Reliability_Mean'] = test_df['Supplier_Name'].map(supplier_reliability_map)

    # Fill NaN with overall mean for unknown suppliers in test
    overall_mean = train_df['Supplier_Reliability'].mean()
    train_df['Supplier_Reliability_Mean'].fillna(overall_mean, inplace=True)
    test_df['Supplier_Reliability_Mean'].fillna(overall_mean, inplace=True)

    # Create supplier frequency features
    supplier_counts = train_df['Supplier_Name'].value_counts().to_dict()
    train_df['Supplier_Frequency'] = train_df['Supplier_Name'].map(supplier_counts)
    test_df['Supplier_Frequency'] = test_df['Supplier_Name'].map(supplier_counts)
    test_df['Supplier_Frequency'].fillna(1, inplace=True)  # Unknown suppliers get frequency 1

    print("Created features: Supplier_Reliability_Mean, Supplier_Frequency")
    return train_df, test_df

def remove_unnecessary_features(train_df, test_df):
    """Remove columns that won't be useful for modeling"""
    print("Removing unnecessary columns...")

    cols_to_remove = [
        'Hospital_Id', 'Supplier_Name', 'Hospital_Location',
        'Order_Placed_Date', 'Delivery_Date'
    ]

    # Remove columns that exist
    for col in cols_to_remove:
        if col in train_df.columns:
            train_df = train_df.drop(col, axis=1)
        if col in test_df.columns:
            test_df = test_df.drop(col, axis=1)

    print(f"Removed columns: {cols_to_remove}")
    return train_df, test_df

def analyze_and_remove_correlations(train_df, test_df, threshold=0.95):
    """Remove highly correlated features"""
    print(f"Analyzing correlations (threshold: {threshold})...")

    # Get numerical columns only
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Transport_Cost' in numeric_cols:
        numeric_cols.remove('Transport_Cost')  # Don't include target in correlation analysis

    # Calculate correlation matrix
    corr_matrix = train_df[numeric_cols].corr().abs()

    # Find highly correlated pairs
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features to drop
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    if to_drop:
        print(f"Removing highly correlated features: {to_drop}")
        train_df = train_df.drop(to_drop, axis=1)
        test_df = test_df.drop([col for col in to_drop if col in test_df.columns], axis=1)
    else:
        print("No highly correlated features found")

    return train_df, test_df

def handle_missing_categoricals(train_df, test_df):
    """Handle missing values in categorical columns"""
    print("Handling missing categorical values...")

    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        if col in train_df.columns:
            # Fill missing with 'Unknown'
            train_df[col].fillna('Unknown', inplace=True)

        if col in test_df.columns:
            test_df[col].fillna('Unknown', inplace=True)

    print(f"Filled missing values with 'Unknown' for: {categorical_cols}")
    return train_df, test_df

def impute_numerical_missing(train_df, test_df):
    """Impute missing numerical values with median"""
    print("Imputing numerical missing values...")

    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Transport_Cost' in numerical_cols:
        numerical_cols.remove('Transport_Cost')  # Don't impute target

    imputation_values = {}

    for col in numerical_cols:
        if train_df[col].isnull().sum() > 0:
            median_val = train_df[col].median()
            imputation_values[col] = median_val

            train_df[col].fillna(median_val, inplace=True)
            if col in test_df.columns:
                test_df[col].fillna(median_val, inplace=True)

    print(f"Imputed columns with median: {list(imputation_values.keys())}")
    return train_df, test_df, imputation_values

def filter_target_outliers(train_df, lower_percentile=2.5, upper_percentile=97.5):
    """Remove outliers from target variable"""
    print(f"Filtering target outliers ({lower_percentile}th to {upper_percentile}th percentile)...")

    if 'Transport_Cost' not in train_df.columns:
        return train_df

    lower_bound = np.percentile(train_df['Transport_Cost'], lower_percentile)
    upper_bound = np.percentile(train_df['Transport_Cost'], upper_percentile)

    original_len = len(train_df)
    train_df = train_df[
        (train_df['Transport_Cost'] >= lower_bound) &
        (train_df['Transport_Cost'] <= upper_bound)
    ].reset_index(drop=True)

    print(f"Removed {original_len - len(train_df)} outliers. Remaining: {len(train_df)} samples")
    return train_df

def encode_categorical_features(train_df, test_df):
    """One-hot encode categorical features"""
    print("One-hot encoding categorical features...")

    # Separate target variable
    if 'Transport_Cost' in train_df.columns:
        y_train = train_df['Transport_Cost'].copy()
        X_train = train_df.drop('Transport_Cost', axis=1)
    else:
        y_train = None
        X_train = train_df.copy()

    X_test = test_df.copy()

    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        # Combine for consistent encoding
        X_train['is_train'] = 1
        X_test['is_train'] = 0
        combined = pd.concat([X_train, X_test], ignore_index=True)

        # One-hot encode
        combined_encoded = pd.get_dummies(combined, columns=categorical_cols, drop_first=False)

        # Split back
        train_mask = combined_encoded['is_train'] == 1
        X_train_encoded = combined_encoded[train_mask].drop('is_train', axis=1)
        X_test_encoded = combined_encoded[~train_mask].drop('is_train', axis=1)

        print(f"Encoded {len(categorical_cols)} categorical columns")
        print(f"Final feature count: {X_train_encoded.shape[1]}")
    else:
        X_train_encoded = X_train
        X_test_encoded = X_test

    return X_train_encoded, X_test_encoded, y_train

def create_polynomial_interactions(X_train, X_test, degree=2):
    """Create polynomial features"""
    print(f"Creating polynomial features (degree={degree})...")

    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    print(f"Feature expansion: {X_train.shape[1]} -> {X_train_poly.shape[1]} features")
    return X_train_poly, X_test_poly, poly

def standardize_features(X_train, X_val, X_test):
    """Standardize features"""
    print("Standardizing features...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def comprehensive_model_search(X_train, y_train, X_val, y_val):
    """Perform comprehensive grid search for both Ridge and Lasso"""
    print("Performing comprehensive model search...")

    # Define parameter grids
    ridge_params = {'alpha': [0.1, 1, 10, 100, 1000, 5000, 10000]}
    lasso_params = {'alpha': [0.01, 0.1, 1, 5, 10, 50, 100]}

    models = {
        'Ridge': {
            'model': Ridge(random_state=42, max_iter=10000),
            'params': ridge_params
        },
        'Lasso': {
            'model': Lasso(random_state=42, max_iter=10000),
            'params': lasso_params
        }
    }

    best_models = {}
    results = {}

    for name, config in models.items():
        print(f"\nSearching {name} parameters...")

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Evaluate on validation set
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)

        results[name] = {
            'best_alpha': grid_search.best_params_['alpha'],
            'cv_rmse': -grid_search.best_score_,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'val_r2': val_r2
        }

        best_models[name] = best_model

        print(f"{name} - Best alpha: {grid_search.best_params_['alpha']}")
        print(f"{name} - CV RMSE: {-grid_search.best_score_:.2f}")
        print(f"{name} - Val RMSE: {val_rmse:.2f}")
        print(f"{name} - Val R²: {val_r2:.4f}")

        if name == 'Lasso':
            n_selected = np.sum(best_model.coef_ != 0)
            print(f"{name} - Features selected: {n_selected}/{len(best_model.coef_)}")

    # Select best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['val_rmse'])
    print(f"\nBest overall model: {best_model_name}")

    return best_models, results, best_model_name

def generate_final_predictions(best_model, X_test, original_test_path='test.csv'):
    """Generate final predictions for submission"""
    print("Generating final predictions...")

    # Load original test for Hospital_Id
    test_original = pd.read_csv(original_test_path)

    # Make predictions
    predictions = best_model.predict(X_test)

    # Create submission
    submission = pd.DataFrame({
        'Hospital_Id': test_original['Hospital_Id'],
        'Transport_Cost': predictions
    })

    print(f"Generated {len(predictions)} predictions")
    print(f"Prediction stats - Mean: {predictions.mean():.2f}, Std: {predictions.std():.2f}")

    return submission

def main_preprocessing_and_modeling():
    """Main function to run the complete pipeline"""
    print("="*80)
    print("COMPREHENSIVE PREPROCESSING AND MODELING PIPELINE")
    print("="*80)

    # Load data
    train_df, test_df = load_and_initial_cleanup()

    # Feature engineering
    train_df, test_df = engineer_date_features(train_df, test_df)
    train_df, test_df = create_supplier_aggregations(train_df, test_df)

    # Data cleaning
    train_df, test_df = remove_unnecessary_features(train_df, test_df)
    train_df, test_df = analyze_and_remove_correlations(train_df, test_df)
    train_df, test_df = handle_missing_categoricals(train_df, test_df)
    train_df, test_df, imputation_values = impute_numerical_missing(train_df, test_df)

    # Outlier removal
    train_df = filter_target_outliers(train_df)

    # Encoding
    X_train, X_test, y_train = encode_categorical_features(train_df, test_df)

    # Train/validation split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"\nFinal data shapes:")
    print(f"X_train: {X_train_split.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")

    # Create polynomial features
    X_train_poly, X_val_poly, poly = create_polynomial_interactions(X_train_split, X_val, degree=2)
    X_test_poly, _, _ = create_polynomial_interactions(X_test, X_test, degree=2)

    # Standardize
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_features(
        X_train_poly, X_val_poly, X_test_poly
    )

    # Model search and training
    best_models, results, best_model_name = comprehensive_model_search(
        X_train_scaled, y_train_split, X_val_scaled, y_val
    )

    # Generate predictions
    final_model = best_models[best_model_name]
    submission = generate_final_predictions(final_model, X_test_scaled)

    # Save submission
    submission.to_csv('submission_comprehensive.csv', index=False)
    print(f"\nSubmission saved to 'submission_comprehensive.csv'")

    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)

    return best_models, results, submission

if __name__ == "__main__":
    models, results, submission = main_preprocessing_and_modeling()
