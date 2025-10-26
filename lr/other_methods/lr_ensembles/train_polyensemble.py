import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
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
        PolynomialFeatures,
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
    SEED = 42
    POLYNOMIAL_DEGREE = 2 # Degree 2 is a good start. Degree 3 can be very slow.

    return POLYNOMIAL_DEGREE, PROCESSED_DATA_PATH, SEED


@app.cell
def _(mean_squared_error, np, r2_score):
    def evaluate_predictions(model_name, y_pred_log, y_val):
        """Helper function to evaluate a set of predictions."""
        print(f"\n--- Evaluating: {model_name} ---")
        y_val_actual = np.expm1(y_val)
        y_pred_actual = np.expm1(y_pred_log)
    
        rmse = np.sqrt(mean_squared_error(y_val_actual, y_pred_actual))
        r2 = r2_score(y_val_actual, y_pred_actual)
    
        print(f"R-squared (R²): {r2:.4f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")

    return (evaluate_predictions,)


@app.cell
def _(
    GridSearchCV,
    KFold,
    Lasso,
    LinearRegression,
    POLYNOMIAL_DEGREE,
    PROCESSED_DATA_PATH,
    PolynomialFeatures,
    Ridge,
    SEED,
    evaluate_predictions,
    np,
    pd,
):
    def main():
        # 1. Load Preprocessed Data
        data = pd.read_pickle(PROCESSED_DATA_PATH)
        X_train, X_val, y_train, y_val = data['X_train'], data['X_val'], data['y_train'], data['y_val']
        print("Data loaded successfully.")

        # 2. Create Polynomial Features
        print(f"\nCreating polynomial features with degree={POLYNOMIAL_DEGREE}...")
        poly = PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False, interaction_only=False)
    
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)
    
        print(f"Feature space expanded from {X_train.shape[1]} to {X_train_poly.shape[1]} features.")

        # Define a reproducible cross-validation strategy
        k_fold = KFold(n_splits=5, shuffle=True, random_state=SEED)

        # --- Model Training on Polynomial Features ---

        # 3. Baseline Linear Regression
        print("\n--- Training: Linear Regression on Poly Features ---")
        lr_poly_model = LinearRegression().fit(X_train_poly, y_train)
        lr_preds = lr_poly_model.predict(X_val_poly)
        evaluate_predictions("Linear Regression (Poly)", lr_preds, y_val)

        # 4. Ridge Regression with Two-Stage Grid Search
        # Stage 1: Broad Search
        print("\n--- Training: Ridge (Poly) - Broad Search ---")
        ridge_params_broad = {'alpha': np.logspace(-3, 3, 7)} # [0.001, ..., 1000]
        ridge_cv_broad = GridSearchCV(Ridge(), ridge_params_broad, cv=k_fold, scoring='neg_mean_squared_error')
        ridge_cv_broad.fit(X_train_poly, y_train)
        best_alpha_ridge = ridge_cv_broad.best_params_['alpha']
        print(f"Broad search best alpha for Ridge: {best_alpha_ridge}")

        # Stage 2: Refined Search
        print("\n--- Training: Ridge (Poly) - Refined Search ---")
        ridge_params_refined = {'alpha': np.linspace(best_alpha_ridge * 0.1, best_alpha_ridge * 2, 10)}
        ridge_cv_refined = GridSearchCV(Ridge(), ridge_params_refined, cv=k_fold, scoring='neg_mean_squared_error')
        ridge_cv_refined.fit(X_train_poly, y_train)
        print(f"Refined search best params for Ridge: {ridge_cv_refined.best_params_}")
        best_ridge_model = ridge_cv_refined.best_estimator_
        ridge_preds = best_ridge_model.predict(X_val_poly)
        evaluate_predictions("Ridge Regression (Poly)", ridge_preds, y_val)

        # 5. Lasso Regression with Two-Stage Grid Search
        # Stage 1: Broad Search
        print("\n--- Training: Lasso (Poly) - Broad Search ---")
        lasso_params_broad = {'alpha': np.logspace(-5, -1, 5)} # [1e-5, ..., 0.1]
        lasso_cv_broad = GridSearchCV(Lasso(max_iter=30000, tol=0.001), lasso_params_broad, cv=k_fold, scoring='neg_mean_squared_error')
        lasso_cv_broad.fit(X_train_poly, y_train)
        best_alpha_lasso = lasso_cv_broad.best_params_['alpha']
        print(f"Broad search best alpha for Lasso: {best_alpha_lasso}")

        # Stage 2: Refined Search
        print("\n--- Training: Lasso (Poly) - Refined Search ---")
        lasso_params_refined = {'alpha': np.linspace(best_alpha_lasso * 0.1, best_alpha_lasso * 2, 10)}
        lasso_cv_refined = GridSearchCV(Lasso(max_iter=30000, tol=0.001), lasso_params_refined, cv=k_fold, scoring='neg_mean_squared_error')
        lasso_cv_refined.fit(X_train_poly, y_train)
        print(f"Refined search best params for Lasso: {lasso_cv_refined.best_params_}")
        best_lasso_model = lasso_cv_refined.best_estimator_
        lasso_preds = best_lasso_model.predict(X_val_poly)
        evaluate_predictions("Lasso Regression (Poly)", lasso_preds, y_val)

        # 6. Ensemble by Averaging
        ensemble_preds = (lr_preds + ridge_preds + lasso_preds) / 3.0
        evaluate_predictions("Averaging Ensemble (Poly)", ensemble_preds, y_val)

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
