import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# --- 1. Advanced Preprocessing Function ---
def preprocess_advanced(train_df, test_df):
    """
    Implements the advanced preprocessing strategy with the KeyError fix.
    """
    print("\n--- Starting Advanced Preprocessing ---")

    # --- Step A: Initial Cleaning on Training Data ONLY ---
    train_clean = train_df[train_df['Transport_Cost'] >= 0].copy()
    lower_bound = train_clean['Transport_Cost'].quantile(0.025)
    upper_bound = train_clean['Transport_Cost'].quantile(0.975)
    train_clean = train_clean[(train_clean['Transport_Cost'] >= lower_bound) & (train_clean['Transport_Cost'] <= upper_bound)]
    print(f"Clipped training data to keep Transport_Cost between {lower_bound:.2f} and {upper_bound:.2f}.")

    y = train_clean['Transport_Cost']
    log_y = np.log1p(y)

    # --- Step B: Learn Parameters from Cleaned Training Data ---
    supplier_reliability_map = train_clean.groupby('Supplier_Name')['Supplier_Reliability'].mean()
    global_reliability_mean = train_clean['Supplier_Reliability'].mean()

    temp_train_numeric = train_clean.select_dtypes(include=np.number).drop(columns=['Transport_Cost'])
    corr_matrix = temp_train_numeric.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    cols_to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]
    print(f"Identified highly correlated features to drop: {cols_to_drop_corr}")

    # --- FIX: Learn medians for ORIGINAL numerical columns ONLY ---
    imputation_medians = {}
    original_num_cols = ['Equipment_Height', 'Equipment_Weight', 'Equipment_Width', 'Base_Transport_Fee']
    for col in original_num_cols:
        imputation_medians[col] = train_clean[col].median()
    print("Learned imputation medians from training data.")

    # --- Step C: Define a Reusable Transformation Function ---
    def transform_data(df, is_test=False):
        df_transformed = df.copy()

        # Date Handling
        df_transformed['Order_Placed_Date'] = pd.to_datetime(df_transformed['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
        df_transformed['Delivery_Date'] = pd.to_datetime(df_transformed['Delivery_Date'], format='%m/%d/%y', errors='coerce')
        swap_mask = df_transformed['Delivery_Date'] < df_transformed['Order_Placed_Date']
        df_transformed.loc[swap_mask, ['Order_Placed_Date', 'Delivery_Date']] = \
            df_transformed.loc[swap_mask, ['Delivery_Date', 'Order_Placed_Date']].values
        df_transformed['Delivery_Time_in_Days'] = (df_transformed['Delivery_Date'] - df_transformed['Order_Placed_Date']).dt.days

        cols_to_drop = ['Hospital_Id', 'Hospital_Location', 'Hospital_Info', 'Order_Placed_Date', 'Delivery_Date']
        if not is_test:
            cols_to_drop.append('Transport_Cost')
        df_transformed.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # Apply learned transformations
        df_transformed['Supplier_Reliability'] = df_transformed['Supplier_Name'].map(supplier_reliability_map).fillna(global_reliability_mean)

        cat_cols_to_fill = ['Rural_Hospital', 'Equipment_Type', 'Transport_Method']
        for col in cat_cols_to_fill:
            df_transformed[col] = df_transformed[col].fillna("Unknown")

        # --- FIX: Apply learned medians to original columns ---
        for col, median_val in imputation_medians.items():
            df_transformed[col] = df_transformed[col].fillna(median_val)

        # --- FIX: Handle imputation for the NEWLY created date column separately ---
        # We can use the median of the column *after* it has been created on the training set.
        # For simplicity and robustness, we can just use the global median learned from the original data.
        delivery_time_median = train_clean.get('Delivery_Time_in_Days', pd.Series(dtype='float64')).median() # Safely get median
        if pd.isna(delivery_time_median): # Fallback if it wasn't pre-calculated
             delivery_time_median = 10 # A reasonable default
        df_transformed['Delivery_Time_in_Days'] = df_transformed['Delivery_Time_in_Days'].fillna(delivery_time_median)

        df_transformed.drop(columns=['Supplier_Name'], inplace=True)
        df_transformed = pd.get_dummies(df_transformed, drop_first=True, dtype=float)
        df_transformed.drop(columns=cols_to_drop_corr, inplace=True, errors='ignore')

        return df_transformed

    # --- Step D: Apply Transformations and Align ---
    X = transform_data(train_clean)
    X_test = transform_data(test_df, is_test=True)

    train_cols = X.columns
    X_test = X_test.reindex(columns=train_cols, fill_value=0)

    print("Advanced preprocessing complete.")
    return X, y, log_y, X_test, test_df['Hospital_Id']

# --- 2. Model Building Function (No changes needed here) ---
def build_model_search():
    """Defines the pipeline with the CORRECT order and GridSearchCV object."""
    pipeline = Pipeline([
        # Step 1: Create polynomial features from original data
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),

        # Step 2: Scale ALL features (original + new polynomial ones)
        ('scaler', StandardScaler()),

        # Step 3: The model
        ('model', Ridge()) # Placeholder
    ])

    param_grid = [
        # The 'poly__degree' parameter is now part of the main pipeline,
        # so we don't need to specify it for each model anymore.
        {'model': [Ridge(random_state=42)], 'model__alpha': [10.0, 100.0, 1000.0]},
        {'model': [Lasso(random_state=42, max_iter=5000)], 'model__alpha': [0.01, 0.1, 1.0]},
        {'model': [ElasticNet(random_state=42, max_iter=5000)], 'model__alpha': [0.1, 1.0], 'model__l1_ratio': [0.1, 0.5, 0.9]}
    ]

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1)
    return grid_search

# --- 3. Main Execution Block (No changes needed here) ---
def main():
    train_df, test_df = pd.read_csv('train.csv'), pd.read_csv('test.csv')
    X, y, log_y, X_test, test_ids = preprocess_advanced(train_df, test_df)

    print("\n--- Experiment 1: Training on Linear Target 'Transport_Cost' ---")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model_search_linear = build_model_search()
    model_search_linear.fit(X_train, y_train)
    best_pipeline_linear = model_search_linear.best_estimator_
    val_preds_linear = best_pipeline_linear.predict(X_val)
    rmse_linear = np.sqrt(mean_squared_error(y_val, val_preds_linear))
    print(f" > Validation RMSE (Linear Target): {rmse_linear:.4f}")
    print(f" > Best Params (Linear Target): {model_search_linear.best_params_}")

    print("\n--- Experiment 2: Training on Log-Transformed Target 'log(Transport_Cost)' ---")
    X_train_log, X_val_log, y_train_log, y_val_log = train_test_split(X, log_y, test_size=0.2, random_state=42)
    model_search_log = build_model_search()
    model_search_log.fit(X_train_log, y_train_log)
    best_pipeline_log = model_search_log.best_estimator_
    val_preds_log = best_pipeline_log.predict(X_val_log)
    val_preds_original_scale = np.expm1(val_preds_log)
    y_val_original_scale = np.expm1(y_val_log)
    rmse_log = np.sqrt(mean_squared_error(y_val_original_scale, val_preds_original_scale))
    print(f" > Validation RMSE (Log Target, converted back to original scale): {rmse_log:.4f}")
    print(f" > Best Params (Log Target): {model_search_log.best_params_}")

    print("\n--- Finalizing and Predicting ---")
    if rmse_log < rmse_linear:
        print("Log-transformed model performed better. Using it for final predictions.")
        final_model = best_pipeline_log
        final_X, final_y, is_log_model = X, log_y, True
    else:
        print("Linearly-scaled model performed better. Using it for final predictions.")
        final_model = best_pipeline_linear
        final_X, final_y, is_log_model = X, y, False

    print("Re-fitting the winning model on the entire training data...")
    final_model.fit(final_X, final_y)

    print("Making final predictions on the test set...")
    final_predictions = final_model.predict(X_test)

    if is_log_model:
        final_predictions = np.expm1(final_predictions)

    submission_df = pd.DataFrame({'Hospital_Id': test_ids, 'Transport_Cost': final_predictions})
    submission_df.to_csv('submission_advanced.csv', index=False)

    print("\n--- Workflow Complete ---")
    print("Submission file 'submission_advanced.csv' has been created.")

if __name__ == '__main__':
    main()
