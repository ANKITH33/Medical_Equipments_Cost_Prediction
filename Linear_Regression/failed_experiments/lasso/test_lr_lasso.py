import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso

    return Lasso, PolynomialFeatures, np, pd


@app.cell
def _():
    # --- Configuration ---
    TRAIN_FILE_PATH = 'train.csv'
    TEST_FILE_PATH = 'test.csv'
    SUBMISSION_FILE_PATH = 'submission_poly_lasso.csv' # New file name for this specific model
    TARGET_VARIABLE = 'Transport_Cost'
    SEED = 42
    POLYNOMIAL_DEGREE = 2

    BEST_LASSO_ALPHA = 0.002 # Replace with the final alpha from your refined Lasso search

    return (
        BEST_LASSO_ALPHA,
        POLYNOMIAL_DEGREE,
        SUBMISSION_FILE_PATH,
        TARGET_VARIABLE,
        TEST_FILE_PATH,
        TRAIN_FILE_PATH,
    )


@app.cell
def _(TARGET_VARIABLE, np, pd):
    # --- Reusable Preprocessing Function (from previous steps) ---
    def preprocess(df, fit_objects=None):
        """A comprehensive preprocessing function for train and test data."""
        if fit_objects is None:
            is_training = True
            fit_objects = {'imputation': {}, 'log_transform_cols': [], 'one_hot_cols': []}
        else:
            is_training = False

        if is_training: df[TARGET_VARIABLE] = df[TARGET_VARIABLE].abs()
        df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], errors='coerce')
        df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')
        df['Delivery_Duration_Days'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days + 1
        df.loc[df['Delivery_Duration_Days'] <= 0, 'Delivery_Duration_Days'] = 1
        df = df.drop(columns=['Hospital_Id', 'Supplier_Name', 'Hospital_Location', 'Order_Placed_Date', 'Delivery_Date'])
    
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if TARGET_VARIABLE in numerical_cols: numerical_cols.remove(TARGET_VARIABLE)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        for col in numerical_cols:
            impute_value = df[col].median() if is_training else fit_objects['imputation'].get(col)
            if is_training: fit_objects['imputation'][col] = impute_value
            df[col] = df[col].fillna(impute_value)
        for col in categorical_cols:
            impute_value = df[col].mode()[0] if is_training else fit_objects['imputation'].get(col)
            if is_training: fit_objects['imputation'][col] = impute_value
            df[col] = df[col].fillna(impute_value)

        skewed_features = ['Equipment_Weight', 'Equipment_Value', 'Base_Transport_Fee', 'Delivery_Duration_Days']
        if is_training: fit_objects['log_transform_cols'] = skewed_features
        for col in fit_objects['log_transform_cols']: df[col] = np.log1p(df[col])
        if is_training: df[TARGET_VARIABLE] = np.log1p(df[TARGET_VARIABLE])
    
        if is_training: fit_objects['one_hot_cols'] = categorical_cols
        df = pd.get_dummies(df, columns=fit_objects['one_hot_cols'], drop_first=True, dtype=float)
        return df, fit_objects

    return (preprocess,)


@app.cell
def _(
    BEST_LASSO_ALPHA,
    Lasso,
    POLYNOMIAL_DEGREE,
    PolynomialFeatures,
    SUBMISSION_FILE_PATH,
    TARGET_VARIABLE,
    TEST_FILE_PATH,
    TRAIN_FILE_PATH,
    np,
    pd,
    preprocess,
):
    def main():
        # 1. Load and Preprocess Data
        print("Loading and preprocessing data...")
        df_train_raw = pd.read_csv(TRAIN_FILE_PATH)
        df_test_raw = pd.read_csv(TEST_FILE_PATH)
        customer_ids = df_test_raw['Hospital_Id']

        df_train_processed, fit_objects = preprocess(df_train_raw.copy())
        df_test_processed, _ = preprocess(df_test_raw.copy(), fit_objects=fit_objects)

        X_train = df_train_processed.drop(columns=[TARGET_VARIABLE])
        y_train = df_train_processed[TARGET_VARIABLE]
        X_test = df_test_processed

        # Align columns before polynomial transformation
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # 2. Apply Polynomial Features Transformation
        print(f"Applying polynomial features (degree={POLYNOMIAL_DEGREE})...")
        poly = PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        print(f"Test features transformed to shape: {X_test_poly.shape}")

        # 3. Train Final Lasso Model on FULL Training Data
        print("\nTraining final Lasso model on all available data...")
        lasso_final = Lasso(alpha=BEST_LASSO_ALPHA, max_iter=30000, tol=0.001).fit(X_train_poly, y_train)
        print("Model training complete.")

        # 4. Predict with the Lasso Model
        print("Making predictions with the final Lasso model...")
        predictions_log = lasso_final.predict(X_test_poly)

        # 5. Format and Save Submission
        actual_predictions = np.expm1(predictions_log)
        actual_predictions[actual_predictions < 0] = 0

        submission_df = pd.DataFrame({'Hospital_Id': customer_ids, 'Transport_Cost': actual_predictions})
        submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)
    
        print(f"\n✅ Submission file '{SUBMISSION_FILE_PATH}' created successfully!")
        print(submission_df.head())

    return (main,)


@app.cell
def _(main):
    main()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
