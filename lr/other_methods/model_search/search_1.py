import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Data Loading Function ---
def load_data(train_path='train.csv', test_path='test.csv'):
    """Loads the training and testing data from CSV files."""
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print("Data loaded successfully.")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure '{train_path}' and '{test_path}' are in the correct directory.")
        return None, None

# --- 2. Preprocessing Function (Leakage-Proof) ---
def preprocess(train_df, test_df):
    """
    Preprocesses the data, learning from the training set and applying to both.
    Implements all agreed-upon rules.
    """
    print("\n--- Starting Preprocessing ---")

    # --- Step A: Clean Training Data & Define Target ---
    train_clean = train_df[train_df['Transport_Cost'] >= 0].copy()
    y = train_clean['Transport_Cost']

    # --- Step B: Learn Parameters ONLY from Training Data ---
    imputation_map = {}

    # Numerical columns: learn MEDIAN
    num_cols = train_clean.select_dtypes(include=np.number).columns.drop('Transport_Cost')
    for col in num_cols:
        imputation_map[col] = train_clean[col].median()

    # Categorical columns: learn MODE
    cat_cols = train_clean.select_dtypes(include=['object']).columns.drop(['Hospital_Id', 'Supplier_Name', 'Hospital_Location'])
    for col in cat_cols:
        imputation_map[col] = train_clean[col].mode()[0]

    print("Learned imputation values from training data.")

    # --- Step C: Define a Transformation Function to Apply to Both Sets ---
    def transform_data(df, is_test=False):
        df_transformed = df.copy()

        # Date Handling: Swap inverted dates
        df_transformed['Order_Placed_Date'] = pd.to_datetime(df_transformed['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
        df_transformed['Delivery_Date'] = pd.to_datetime(df_transformed['Delivery_Date'], format='%m/%d/%y', errors='coerce')

        swap_mask = df_transformed['Delivery_Date'] < df_transformed['Order_Placed_Date']
        df_transformed.loc[swap_mask, ['Order_Placed_Date', 'Delivery_Date']] = \
            df_transformed.loc[swap_mask, ['Delivery_Date', 'Order_Placed_Date']].values

        df_transformed['Delivery_Time_in_Days'] = (df_transformed['Delivery_Date'] - df_transformed['Order_Placed_Date']).dt.days

        # Drop unnecessary columns
        cols_to_drop = ['Hospital_Id', 'Supplier_Name', 'Hospital_Location', 'Hospital_Info', 'Order_Placed_Date', 'Delivery_Date']
        if not is_test:
            cols_to_drop.append('Transport_Cost')
        df_transformed.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # Apply learned imputations
        for col, value in imputation_map.items():
            if col in df_transformed.columns:
                df_transformed[col] = df_transformed[col].fillna(value)

        # Encoding
        df_transformed = pd.get_dummies(df_transformed, drop_first=True, dtype=float)

        return df_transformed

    # --- Step D: Apply Transformations ---
    X = transform_data(train_clean)
    X_test = transform_data(test_df, is_test=True)

    # Align columns to ensure consistency
    train_cols = X.columns
    X_test = X_test.reindex(columns=train_cols, fill_value=0)

    print("Preprocessing complete. Train and test sets are ready.")
    return X, y, X_test, test_df['Hospital_Id']

# --- 3. Model Building Function ---
def build_model_search():
    """Defines the pipeline and GridSearchCV object for model searching."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(include_bias=False)),
        ('model', Ridge()) # Placeholder model
    ])

    param_grid = [
        {
            'poly__degree': [2],
            'model': [Ridge(random_state=42)],
            'model__alpha': [10.0, 100.0, 1000.0, 10000.0, 100000.0]
        },
        {
            'poly__degree': [2],
            'model': [Lasso(random_state=42, max_iter=5000)],
            'model__alpha': [0.01, 0.1, 1.0]
        },
        {
            'poly__degree': [2],
            'model': [ElasticNet(random_state=42, max_iter=5000)],
            'model__alpha': [0.1, 1.0],
            'model__l1_ratio': [0.1, 0.5, 0.9]
        }
    ]

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )
    return grid_search

# --- 4. Main Execution Block ---
def main():
    """Orchestrates the entire model training and prediction workflow."""
    train_df, test_df = load_data()
    if train_df is None:
        return

    X, y, X_test, test_ids = preprocess(train_df, test_df)

    # Create a validation set to check performance before final prediction
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model_searcher = build_model_search()

    print("\n--- Training Model Search on the training set ---")
    model_searcher.fit(X_train, y_train)

    best_pipeline = model_searcher.best_estimator_

    print("\n--- Validation Results ---")
    print(f" > Best params found: {model_searcher.best_params_}")

    val_preds = best_pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    r2 = r2_score(y_val, val_preds)

    print(f" > Best Model Validation RMSE: {rmse:.4f}")
    print(f" > Best Model Validation R²:   {r2:.4f}")

    print("\n--- Finalizing Model ---")
    print("Re-fitting the best model on the ENTIRE training data (X, y)...")
    best_pipeline.fit(X, y)

    print("Making final predictions on the test set...")
    final_predictions = best_pipeline.predict(X_test)

    # Create submission file
    submission_df = pd.DataFrame({'Hospital_Id': test_ids, 'Transport_Cost': final_predictions})
    submission_df.to_csv('submission.csv', index=False)

    print("\n--- Workflow Complete ---")
    print("Submission file 'submission.csv' has been created.")
    print("First 5 rows of submission file:")
    print(submission_df.head())

if __name__ == '__main__':
    main()
