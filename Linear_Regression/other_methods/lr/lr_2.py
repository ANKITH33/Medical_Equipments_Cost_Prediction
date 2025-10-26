import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.metrics import mean_squared_error, r2_score
    import warnings

    warnings.filterwarnings('ignore', category=FutureWarning)

    return (
        GridSearchCV,
        KFold,
        Lasso,
        LinearRegression,
        Ridge,
        mean_squared_error,
        np,
        pd,
        r2_score,
    )


@app.cell
def _():
    # --- Configuration ---
    PROCESSED_DATA_PATH = 'processed_data.pkl'
    SEED = 42 # The seed for reproducibility

    return PROCESSED_DATA_PATH, SEED


@app.cell
def _(mean_squared_error, np, r2_score):
    def evaluate_model(model_name, model, X_val, y_val):
        print(f"\n--- Evaluating: {model_name} ---")
        y_pred_log = model.predict(X_val)
        y_val_actual = np.expm1(y_val)
        y_pred_actual = np.expm1(y_pred_log)
    
        rmse = np.sqrt(mean_squared_error(y_val_actual, y_pred_actual))
        r2 = r2_score(y_val_actual, y_pred_actual)
    
        print(f"R-squared (R²): {r2:.4f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    
        if hasattr(model, 'best_params_'):
            print(f"Best Hyperparameters: {model.best_params_}")

    return (evaluate_model,)


@app.cell
def _(
    GridSearchCV,
    KFold,
    Lasso,
    LinearRegression,
    Ridge,
    evaluate_model,
    np,
    pd,
):
    def run_models(data_path, random_seed):
        # 1. Load Data
        data = pd.read_pickle(data_path)
        X_train, X_val, y_train, y_val = data['X_train'], data['X_val'], data['y_train'], data['y_val']
        print("Data loaded successfully.")

        # Define a reproducible cross-validation strategy
        k_fold = KFold(n_splits=5, shuffle=True, random_state=random_seed)

        # 2. Baseline Linear Regression (no change needed)
        lr_model = LinearRegression().fit(X_train, y_train)
        evaluate_model("Linear Regression (Baseline)", lr_model, X_val, y_val)

        # 3. Ridge Regression with REFINED Grid Search
        # Previous best was 1.0. Let's search finely around it.
        print("\n--- Training: Ridge Regression (Refined Search) ---")
        ridge_params = {'alpha': np.linspace(0.5, 1.5, 11)} # e.g., [0.5, 0.6, ..., 1.4, 1.5]
        ridge_cv = GridSearchCV(Ridge(), ridge_params, cv=k_fold, scoring='neg_mean_squared_error')
        ridge_cv.fit(X_train, y_train)
        evaluate_model("Ridge Regression", ridge_cv, X_val, y_val)

        # 4. Lasso Regression with REFINED Grid Search
        # Previous best was 0.0001. Let's search finely around it.
        print("\n--- Training: Lasso Regression (Refined Search) ---")
        lasso_params = {'alpha': np.linspace(0.00005, 0.0005, 10)} # e.g., [5e-5, ..., 5e-4]
        lasso_cv = GridSearchCV(Lasso(max_iter=20000), lasso_params, cv=k_fold, scoring='neg_mean_squared_error')
        lasso_cv.fit(X_train, y_train)
        evaluate_model("Lasso Regression", lasso_cv, X_val, y_val)
    
        best_lasso = lasso_cv.best_estimator_
        zero_coefs = np.sum(best_lasso.coef_ == 0)
        print(f"\nLasso eliminated {zero_coefs} out of {len(best_lasso.coef_)} features.")

    return (run_models,)


@app.cell
def _(PROCESSED_DATA_PATH, SEED, run_models):
    run_models(PROCESSED_DATA_PATH, SEED)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
