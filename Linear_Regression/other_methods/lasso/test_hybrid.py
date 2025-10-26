import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import Pipeline

    return (
        Lasso,
        MinMaxScaler,
        Pipeline,
        PolynomialFeatures,
        SelectKBest,
        f_regression,
        np,
        pd,
    )


@app.cell
def _():
    # --- Configuration ---
    TRAIN_FILE_PATH = 'train.csv'
    TEST_FILE_PATH = 'test.csv'
    SUBMISSION_FILE_PATH = 'submission_hybrid.csv'
    TARGET_VARIABLE = 'Transport_Cost'
    POLYNOMIAL_DEGREE = 2
    NUM_FEATURES_TO_SELECT = 100

    # IMPORTANT: UPDATE THIS VALUE after running train_hybrid_approach.py
    BEST_LASSO_ALPHA = 1e-05 # Replace with the alpha from the training script's output

    return (
        BEST_LASSO_ALPHA,
        NUM_FEATURES_TO_SELECT,
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
    MinMaxScaler,
    NUM_FEATURES_TO_SELECT,
    POLYNOMIAL_DEGREE,
    Pipeline,
    PolynomialFeatures,
    SUBMISSION_FILE_PATH,
    SelectKBest,
    TARGET_VARIABLE,
    TEST_FILE_PATH,
    TRAIN_FILE_PATH,
    f_regression,
    np,
    pd,
):
    def main():
        # 1. Load Data
        # CORRECTED DATE PARSING
        df_train_raw = pd.read_csv(TRAIN_FILE_PATH, parse_dates=['Order_Placed_Date', 'Delivery_Date'], date_format='%m/%d/%y')
        df_test_raw = pd.read_csv(TEST_FILE_PATH, parse_dates=['Order_Placed_Date', 'Delivery_Date'], date_format='%m/%d/%y')
        customer_ids = df_test_raw['Hospital_Id']

        # --- Define Reusable Processing Steps ---
        def process_data(df, is_train=True, fit_objects=None):
            if is_train:
                fit_objects = {'imputation': {}, 'one_hot_cols': []}
                df[TARGET_VARIABLE] = df[TARGET_VARIABLE].abs()
        
            # CORRECTED DURATION CALCULATION
            df['Delivery_Duration_Days'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days + 1
            df.loc[df['Delivery_Duration_Days'] <= 0, 'Delivery_Duration_Days'] = 1
        
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
        
            # Manual Features
            df['Equipment_Height'] = df['Equipment_Height'].replace(0, 1)
            df['Equipment_Width'] = df['Equipment_Width'].replace(0, 1)
            df['Equipment_Volume'] = df['Equipment_Height'] * df['Equipment_Width']
            df['Value_Density'] = df['Equipment_Value'] / df['Equipment_Volume']
            df['Weight_Density'] = df['Equipment_Weight'] / df['Equipment_Volume']
            df['Service_Level_Score'] = (df['Urgent_Shipping'] == 'Yes').astype(int) + (df['Installation_Service'] == 'Yes').astype(int) + (df['CrossBorder_Shipping'] == 'Yes').astype(int)
            df.replace([np.inf, -np.inf], 0, inplace=True)
        
            df = df.drop(columns=['Order_Placed_Date', 'Delivery_Date', 'Hospital_Id', 'Supplier_Name', 'Hospital_Location'])
        
            final_categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if is_train: fit_objects['one_hot_cols'] = final_categorical_cols
            df = pd.get_dummies(df, columns=fit_objects['one_hot_cols'], drop_first=True, dtype=float)
        
            # Log and Log-of-Log
            skewed_features = ['Equipment_Weight', 'Equipment_Value', 'Base_Transport_Fee', 'Delivery_Duration_Days', 'Equipment_Volume', 'Value_Density', 'Weight_Density']
            for col in skewed_features:
                if col in df.columns: df[col] = np.log1p(df[col])
            for col in skewed_features:
                if col in df.columns: df[col] = np.log1p(df[col])
            
            if is_train: df[TARGET_VARIABLE] = np.log1p(df[TARGET_VARIABLE])
        
            return df, fit_objects

        # --- Execute Pipeline ---
        df_train, fit_objects = process_data(df_train_raw.copy(), is_train=True)
        X_train = df_train.drop(columns=[TARGET_VARIABLE])
        y_train = df_train[TARGET_VARIABLE]

        df_test, _ = process_data(df_test_raw.copy(), is_train=False, fit_objects=fit_objects)
        X_test = df_test.reindex(columns=X_train.columns, fill_value=0)

        # 2. Define and Fit the Final Pipeline on ALL Training Data
        print("Training final pipeline on all data...")
        final_pipeline = Pipeline([
            ('normalizer', MinMaxScaler()),
            ('poly', PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False)),
            ('selector', SelectKBest(f_regression, k=NUM_FEATURES_TO_SELECT)),
            ('model', Lasso(alpha=BEST_LASSO_ALPHA, max_iter=30000, tol=0.001))
        ])
        final_pipeline.fit(X_train, y_train)

        # 3. Predict and Save
        print("Predicting and saving submission...")
        predictions_log = final_pipeline.predict(X_test)
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
