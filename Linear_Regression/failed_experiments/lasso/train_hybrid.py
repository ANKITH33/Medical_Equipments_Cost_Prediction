import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, KFold, GridSearchCV
    from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error
    import warnings

    warnings.filterwarnings('ignore')

    return (
        GridSearchCV,
        KFold,
        Lasso,
        MinMaxScaler,
        Pipeline,
        PolynomialFeatures,
        SelectKBest,
        f_regression,
        mean_squared_error,
        np,
        pd,
        train_test_split,
    )


@app.cell
def _():
    # --- Configuration ---
    TRAIN_FILE_PATH = 'train.csv'
    TARGET_VARIABLE = 'Transport_Cost'
    SEED = 42
    POLYNOMIAL_DEGREE = 2
    NUM_FEATURES_TO_SELECT = 100

    return (
        NUM_FEATURES_TO_SELECT,
        POLYNOMIAL_DEGREE,
        SEED,
        TARGET_VARIABLE,
        TRAIN_FILE_PATH,
    )


@app.cell
def _(
    GridSearchCV,
    KFold,
    Lasso,
    MinMaxScaler,
    NUM_FEATURES_TO_SELECT,
    POLYNOMIAL_DEGREE,
    Pipeline,
    PolynomialFeatures,
    SEED,
    SelectKBest,
    TARGET_VARIABLE,
    TRAIN_FILE_PATH,
    f_regression,
    mean_squared_error,
    np,
    pd,
    train_test_split,
):
    def main():
        # 1. Load and Preprocess
        print("Step 1: Loading and initial cleaning...")
        # CORRECTED DATE PARSING: Pass a list of column names
        df = pd.read_csv(TRAIN_FILE_PATH, 
                         parse_dates=['Order_Placed_Date', 'Delivery_Date'], 
                         date_format='%m/%d/%y')
    
        df[TARGET_VARIABLE] = df[TARGET_VARIABLE].abs()
    
        # CORRECTED DURATION CALCULATION: Use the original column names
        df['Delivery_Duration_Days'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days + 1
        df.loc[df['Delivery_Duration_Days'] <= 0, 'Delivery_Duration_Days'] = 1
    
        # 2. Imputation
        print("Step 2: Imputing missing values...")
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        numerical_cols.remove(TARGET_VARIABLE)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in numerical_cols: df[col] = df[col].fillna(df[col].median())
        for col in categorical_cols: df[col] = df[col].fillna(df[col].mode()[0])

        # 3. Manual Feature Engineering
        print("Step 3: Creating manual features (Volume, Density)...")
        df['Equipment_Height'] = df['Equipment_Height'].replace(0, 1)
        df['Equipment_Width'] = df['Equipment_Width'].replace(0, 1)
        df['Equipment_Volume'] = df['Equipment_Height'] * df['Equipment_Width']
        df['Value_Density'] = df['Equipment_Value'] / df['Equipment_Volume']
        df['Weight_Density'] = df['Equipment_Weight'] / df['Equipment_Volume']
        df['Service_Level_Score'] = (df['Urgent_Shipping'] == 'Yes').astype(int) + (df['Installation_Service'] == 'Yes').astype(int) + (df['CrossBorder_Shipping'] == 'Yes').astype(int)
        df.replace([np.inf, -np.inf], 0, inplace=True)
    
        # 4. Drop unused columns and Encode
        print("Step 4: Dropping unused columns and one-hot encoding...")
        df = df.drop(columns=['Order_Placed_Date', 'Delivery_Date', 'Hospital_Id', 'Supplier_Name', 'Hospital_Location'])
        final_categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        df = pd.get_dummies(df, columns=final_categorical_cols, drop_first=True, dtype=float)

        # 5. Define features for transformation
        skewed_features = ['Equipment_Weight', 'Equipment_Value', 'Base_Transport_Fee', 'Delivery_Duration_Days', 'Equipment_Volume', 'Value_Density', 'Weight_Density']
    
        # 6. First Log Transform
        print("Step 6: Applying first log transform...")
        for col in skewed_features:
            if col in df.columns: df[col] = np.log1p(df[col])
        
        # 7. Second Log Transform (Log-of-Log)
        print("Step 7: Applying second log transform (log-of-log)...")
        for col in skewed_features:
            if col in df.columns: df[col] = np.log1p(df[col])
        
        df[TARGET_VARIABLE] = np.log1p(df[TARGET_VARIABLE])

        # 8. Split Data
        X = df.drop(columns=[TARGET_VARIABLE])
        y = df[TARGET_VARIABLE]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

        # 9. Create the Full Pipeline
        pipeline = Pipeline([
            ('normalizer', MinMaxScaler()),
            ('poly', PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False)),
            ('selector', SelectKBest(f_regression, k=NUM_FEATURES_TO_SELECT)),
            ('model', Lasso(max_iter=30000, tol=0.001))
        ])

        # 10. Define Hyperparameter Grid and Train
        params = {'model__alpha': np.logspace(-5, -1, 20)}
        print("\nStep 10: Training full pipeline with GridSearchCV...")
        k_fold = KFold(n_splits=5, shuffle=True, random_state=SEED)
        grid_search = GridSearchCV(pipeline, params, cv=k_fold, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
    
        best_model_pipeline = grid_search.best_estimator_
        print(f"Best alpha found for Lasso: {grid_search.best_params_['model__alpha']}")

        # 11. Evaluate
        y_pred_log = best_model_pipeline.predict(X_val)
        y_val_actual = np.expm1(y_val)
        y_pred_actual = np.expm1(y_pred_log)
        rmse = np.sqrt(mean_squared_error(y_val_actual, y_pred_actual))
    
        print(f"\nValidation RMSE with Hybrid Approach: ${rmse:,.2f}")

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
