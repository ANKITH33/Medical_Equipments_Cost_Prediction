import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import Lasso
    import pickle

    return Lasso, PolynomialFeatures, StandardScaler, np, pd


@app.cell
def _():
    # --- Configuration ---
    TRAIN_FILE_PATH = 'train.csv'
    TEST_FILE_PATH = 'test.csv'
    SUBMISSION_FILE_PATH = 'submission_final_pipeline.csv'
    TARGET_VARIABLE = 'Transport_Cost'
    POLYNOMIAL_DEGREE = 2

    # IMPORTANT: UPDATE THIS VALUE after running train_final_pipeline.py
    BEST_LASSO_ALPHA = 0.002 # Replace with the alpha from the new training script's output

    return (
        BEST_LASSO_ALPHA,
        POLYNOMIAL_DEGREE,
        SUBMISSION_FILE_PATH,
        TARGET_VARIABLE,
        TEST_FILE_PATH,
        TRAIN_FILE_PATH,
    )


@app.cell
def _(
    BEST_LASSO_ALPHA,
    Lasso,
    POLYNOMIAL_DEGREE,
    PolynomialFeatures,
    SUBMISSION_FILE_PATH,
    StandardScaler,
    TARGET_VARIABLE,
    TEST_FILE_PATH,
    TRAIN_FILE_PATH,
    np,
    pd,
):
    def main():
        # 1. Load Data
        df_train_raw = pd.read_csv(TRAIN_FILE_PATH)
        df_test_raw = pd.read_csv(TEST_FILE_PATH)
        customer_ids = df_test_raw['Hospital_Id']

        # --- Define Reusable Processing Steps ---
        # These steps must be applied identically to train and test data
    
        # A: Impute, Engineer, Encode
        def initial_process(df, is_train=True, fit_objects=None):
            if is_train:
                fit_objects = {'imputation': {}, 'one_hot_cols': []}
                df[TARGET_VARIABLE] = df[TARGET_VARIABLE].abs()
        
            df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], errors='coerce')
            df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')
            df['Delivery_Duration_Days'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days + 1
            df.loc[df['Delivery_Duration_Days'] <= 0, 'Delivery_Duration_Days'] = 1
            df = df.drop(columns=['Hospital_Id', 'Supplier_Name', 'Hospital_Location', 'Order_Placed_Date', 'Delivery_Date'])
        
            numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
            if TARGET_VARIABLE in numerical_cols: numerical_cols.remove(TARGET_VARIABLE)
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

            for col in numerical_cols:
                impute_value = df[col].median() if is_train else fit_objects['imputation'].get(col)
                if is_train: fit_objects['imputation'][col] = impute_value
                df[col] = df[col].fillna(impute_value)
            for col in categorical_cols:
                impute_value = df[col].mode()[0] if is_train else fit_objects['imputation'].get(col)
                if is_train: fit_objects['imputation'][col] = impute_value
                df[col] = df[col].fillna(impute_value)
        
            if is_train: fit_objects['one_hot_cols'] = categorical_cols
            df = pd.get_dummies(df, columns=fit_objects['one_hot_cols'], drop_first=True, dtype=float)
            return df, fit_objects

        # B: Log Transform
        def log_transform(df, is_train=True):
            skewed_features = ['Equipment_Weight', 'Equipment_Value', 'Base_Transport_Fee', 'Delivery_Duration_Days']
            for col in skewed_features:
                if col in df.columns: df[col] = np.log1p(df[col])
            if is_train: df[TARGET_VARIABLE] = np.log1p(df[TARGET_VARIABLE])
            return df

        # --- Execute Pipeline ---
        # Process Train Data
        df_train, fit_objects = initial_process(df_train_raw.copy(), is_train=True)
        df_train = log_transform(df_train, is_train=True)
        X_train = df_train.drop(columns=[TARGET_VARIABLE])
        y_train = df_train[TARGET_VARIABLE]

        # Process Test Data
        df_test, _ = initial_process(df_test_raw.copy(), is_train=False, fit_objects=fit_objects)
        df_test = log_transform(df_test, is_train=False)
        X_test = df_test.reindex(columns=X_train.columns, fill_value=0)

        # Scale Data
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Polynomial Features
        poly = PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False).fit(X_train_scaled)
        X_train_poly = poly.transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)

        # Train Final Model
        print("Training final model on all data...")
        final_model = Lasso(alpha=BEST_LASSO_ALPHA, max_iter=40000, tol=0.001).fit(X_train_poly, y_train)

        # Predict and Save
        print("Predicting and saving submission...")
        predictions_log = final_model.predict(X_test_poly)
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
