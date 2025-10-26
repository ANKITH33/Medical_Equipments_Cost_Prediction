import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, KFold, GridSearchCV
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import Lasso
    from sklearn.metrics import mean_squared_error
    import warnings

    warnings.filterwarnings('ignore', category=FutureWarning)

    return (
        GridSearchCV,
        KFold,
        Lasso,
        PolynomialFeatures,
        StandardScaler,
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

    return POLYNOMIAL_DEGREE, SEED, TARGET_VARIABLE, TRAIN_FILE_PATH


@app.cell
def _(
    GridSearchCV,
    KFold,
    Lasso,
    POLYNOMIAL_DEGREE,
    PolynomialFeatures,
    SEED,
    StandardScaler,
    TARGET_VARIABLE,
    TRAIN_FILE_PATH,
    mean_squared_error,
    np,
    pd,
    train_test_split,
):
    def main():
        # 1. Load Data
        df = pd.read_csv(TRAIN_FILE_PATH)

        # --- Start of Full Preprocessing Pipeline ---
        # Step A: Initial Cleaning & Feature Engineering
        df[TARGET_VARIABLE] = df[TARGET_VARIABLE].abs()
        df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], errors='coerce')
        df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')
        df['Delivery_Duration_Days'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days + 1
        df.loc[df['Delivery_Duration_Days'] <= 0, 'Delivery_Duration_Days'] = 1
        df = df.drop(columns=['Hospital_Id', 'Supplier_Name', 'Hospital_Location', 'Order_Placed_Date', 'Delivery_Date'])

        # Step B: Imputation
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        numerical_cols.remove(TARGET_VARIABLE)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in numerical_cols: df[col] = df[col].fillna(df[col].median())
        for col in categorical_cols: df[col] = df[col].fillna(df[col].mode()[0])

        # Step C: One-Hot Encoding
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=float)

        # Step D: Log Transformation (on skewed features AND target)
        skewed_features = ['Equipment_Weight', 'Equipment_Value', 'Base_Transport_Fee', 'Delivery_Duration_Days']
        for col in skewed_features:
            if col in df.columns: df[col] = np.log1p(df[col])
        df[TARGET_VARIABLE] = np.log1p(df[TARGET_VARIABLE])
        # --- End of Initial Preprocessing ---

        # 2. Split Data into Train/Validation
        X = df.drop(columns=[TARGET_VARIABLE])
        y = df[TARGET_VARIABLE]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

        # 3. Scaling (Fit on Train, Transform Both)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 4. Polynomial Features (Fit on Train, Transform Both)
        poly = PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_val_poly = poly.transform(X_val_scaled)
        print(f"Final feature space has {X_train_poly.shape[1]} features.")

        # 5. Train Model with Hyperparameter Tuning
        print("\nTraining Lasso model on fully processed data...")
        k_fold = KFold(n_splits=5, shuffle=True, random_state=SEED)
        lasso_params = {'alpha': np.logspace(-4, 0, 10)} # Search from 0.0001 to 1
        lasso_cv = GridSearchCV(Lasso(max_iter=40000, tol=0.001), lasso_params, cv=k_fold, scoring='neg_mean_squared_error')
        lasso_cv.fit(X_train_poly, y_train)
    
        best_model = lasso_cv.best_estimator_
        print(f"Best alpha found: {best_model.alpha}")

        # 6. Evaluate on Validation Set
        y_pred_log = best_model.predict(X_val_poly)
        y_val_actual = np.expm1(y_val)
        y_pred_actual = np.expm1(y_pred_log)
        rmse = np.sqrt(mean_squared_error(y_val_actual, y_pred_actual))
    
        print(f"\nValidation RMSE on final pipeline: ${rmse:,.2f}")
        print("This score should be much more reasonable.")

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
