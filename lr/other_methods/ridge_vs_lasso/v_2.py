import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(21)

def train_ridge_lasso_with_gridsearch():
    """
    Train Ridge and Lasso models with polynomial features and grid search for best alpha
    """

    print("Loading preprocessed data...")
    X_train = pd.read_csv('X_train_preprocessed.csv')
    y_train = pd.read_csv('y_train.csv')
    X_test = pd.read_csv('X_test_preprocessed.csv')

    # Convert y_train to series if it's a DataFrame
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    print(f"Original dataset shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")

    # Check for any remaining missing values
    train_missing = X_train.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()
    print(f"\nMissing values - Train: {train_missing}, Test: {test_missing}")

    # Split train into train and validation (85% train, 15% validation)
    print(f"\nSplitting into train/validation (85%/15%)...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=23, shuffle=True
    )

    print(f"After split:")
    print(f"X_train_split: {X_train_split.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_train_split: {y_train_split.shape}")
    print(f"y_val: {y_val.shape}")

    # Create polynomial features
    print(f"\nCreating polynomial features (degree=3)...")
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)

    print("Fitting polynomial transformer on training data...")
    X_train_poly = poly.fit_transform(X_train_split)

    print("Transforming validation and test data...")
    X_val_poly = poly.transform(X_val)
    X_test_poly = poly.transform(X_test)

    print(f"After polynomial features:")
    print(f"X_train_poly: {X_train_poly.shape}")
    print(f"X_val_poly: {X_val_poly.shape}")
    print(f"X_test_poly: {X_test_poly.shape}")
    print(f"Feature expansion: {X_train.shape[1]} -> {X_train_poly.shape[1]} features")

    # Standardize features (after polynomial transformation)
    print(f"\nStandardizing polynomial features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_val_scaled = scaler.transform(X_val_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    print(f"Features standardized using StandardScaler")

    # Define alpha ranges for grid search
    ridge_alphas = [0.1, 1, 10, 100, 1000, 5000]
    lasso_alphas = [0.1, 1, 5, 10, 50, 100, 500]

    print(f"\nGrid Search Parameters:")
    print(f"Ridge alphas: {ridge_alphas}")
    print(f"Lasso alphas: {lasso_alphas}")

    # Define models with parameter grids
    models_params = {
        'Ridge': {
            'model': Ridge(random_state=2, max_iter=5000),
            'params': {'alpha': ridge_alphas}
        },
        'Lasso': {
            'model': Lasso(random_state=2, max_iter=5000),
            'params': {'alpha': lasso_alphas}
        }
    }

    best_models = {}
    all_results = {}

    print(f"\nPerforming Grid Search...")
    print("="*80)

    for name, model_info in models_params.items():
        print(f"\nGrid Search for {name}...")

        # Perform grid search with 5-fold CV
        grid_search = GridSearchCV(
            model_info['model'],
            model_info['params'],
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=2
        )

        # Fit grid search
        grid_search.fit(X_train_scaled, y_train_split)

        # Get best model
        best_model = grid_search.best_estimator_
        best_models[name] = best_model

        print(f"Best {name} alpha: {grid_search.best_params_['alpha']}")
        print(f"Best {name} CV RMSE: {-grid_search.best_score_:.2f}")

        # Store detailed results
        results_df = pd.DataFrame(grid_search.cv_results_)
        all_results[name] = results_df

        # Print all alpha results
        print(f"\n{name} - All Alpha Results:")
        print("-" * 50)
        for i, alpha in enumerate(model_info['params']['alpha']):
            mean_score = -results_df['mean_test_score'].iloc[i]
            std_score = results_df['std_test_score'].iloc[i]
            print(f"Alpha {alpha:8.1f}: CV RMSE = {mean_score:8.2f} (+/- {std_score*2:.2f})")

        # Evaluate best model on validation set
        y_train_pred = best_model.predict(X_train_scaled)
        y_val_pred = best_model.predict(X_val_scaled)

        # Calculate metrics
        train_mse = mean_squared_error(y_train_split, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train_split, y_train_pred)
        train_r2 = r2_score(y_train_split, y_train_pred)

        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = np.sqrt(val_mse)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        print(f"\nBest {name} Performance on Train/Val Split:")
        print(f"  Train - MSE: {train_mse:.2f}, RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.4f}")
        print(f"  Val   - MSE: {val_mse:.2f}, RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}")

        # Store final results
        all_results[f'{name}_final'] = {
            'best_alpha': grid_search.best_params_['alpha'],
            'cv_rmse': -grid_search.best_score_,
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2
        }

        # Show feature selection info for Lasso
        if name == 'Lasso':
            n_features_selected = np.sum(best_model.coef_ != 0)
            print(f"  Features selected by best Lasso: {n_features_selected}/{len(best_model.coef_)}")

    # Compare best models
    print(f"\n" + "="*80)
    print("FINAL MODEL COMPARISON:")
    print("="*80)

    best_model_name = None
    best_val_rmse = float('inf')

    for name in ['Ridge', 'Lasso']:
        metrics = all_results[f'{name}_final']
        print(f"\nBest {name} (alpha={metrics['best_alpha']}):")
        print(f"  CV RMSE: {metrics['cv_rmse']:.2f}")
        print(f"  Validation RMSE: {metrics['val_rmse']:.2f}")
        print(f"  Validation R²: {metrics['val_r2']:.4f}")

        if metrics['val_rmse'] < best_val_rmse:
            best_val_rmse = metrics['val_rmse']
            best_model_name = name

    print(f"\nOverall Best Model: {best_model_name} (Validation RMSE: {best_val_rmse:.2f})")

    # Get the overall best model
    final_best_model = best_models[best_model_name]

    # Load original test data to get Hospital_Id
    print(f"\nLoading original test data for Hospital_Id...")
    test_original = pd.read_csv('test.csv')

    # Make predictions on test set
    print(f"Making predictions on test set with best {best_model_name}...")
    test_predictions = final_best_model.predict(X_test_scaled)

    # Create submission DataFrame
    submission = pd.DataFrame({
        'Hospital_Id': test_original['Hospital_Id'],
        'Transport_Cost': test_predictions
    })

    print(f"\nSubmission file created:")
    print(f"Shape: {submission.shape}")
    print(f"Columns: {submission.columns.tolist()}")
    print(f"\nFirst few predictions:")
    print(submission.head())

    print(f"\nPrediction statistics:")
    print(f"Mean: {test_predictions.mean():.2f}")
    print(f"Std: {test_predictions.std():.2f}")
    print(f"Min: {test_predictions.min():.2f}")
    print(f"Max: {test_predictions.max():.2f}")

    # Save submission
    submission.to_csv('submission_gridsearch.csv', index=False)
    print(f"\nSubmission saved to 'submission_gridsearch.csv'")

    # Feature importance for Ridge (top 20 features)
    if best_model_name == 'Ridge':
        print(f"\nTop 20 features by absolute coefficient value (Best Ridge):")
        feature_names = poly.get_feature_names_out(X_train.columns)
        coef_abs = np.abs(final_best_model.coef_)
        top_indices = np.argsort(coef_abs)[-20:][::-1]

        for i, idx in enumerate(top_indices, 1):
            print(f"{i:2d}. {feature_names[idx]}: {final_best_model.coef_[idx]:.4f}")

    return all_results, best_models, best_model_name, submission, poly, scaler

# Run the training
if __name__ == "__main__":
    print("Starting Ridge and Lasso model training with Grid Search...")
    print("="*80)

    results, models, best_model, submission, poly_transformer, scaler = train_ridge_lasso_with_gridsearch()

    print("\n" + "="*80)
    print("GRID SEARCH TRAINING COMPLETE!")
    print("="*80)

    print(f"\nFinal Results Summary:")
    print(f"Best model: {best_model}")
    print(f"Polynomial degree: 3")
    print(f"Original features: 36")
    print(f"Polynomial features: {poly_transformer.n_output_features_}")
    print(f"Submission file: submission_gridsearch.csv ({submission.shape[0]} predictions)")

    print(f"\nFiles created:")
    print(f"- submission_gridsearch.csv")

    print("\nGrid search training completed successfully!")
