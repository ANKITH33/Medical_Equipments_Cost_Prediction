import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(456)

def load_processed_datasets():
    """
    Load the preprocessed datasets from the v5 preprocessing pipeline.
    """
    print("Loading preprocessed datasets...")
    X_train = pd.read_csv('X_train_processed_v5.csv')
    y_train = pd.read_csv('y_train_processed_v5.csv')
    X_test = pd.read_csv('X_test_processed_v5.csv')
    test_ids = pd.read_csv('test_ids_v5.csv')

    # Convert y_train to series if it's a DataFrame
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    # Convert test_ids to series if it's a DataFrame
    if isinstance(test_ids, pd.DataFrame):
        test_ids = test_ids.iloc[:, 0]

    print(f"Dataset shapes loaded:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"test_ids: {test_ids.shape}")

    return X_train, y_train, X_test, test_ids

def create_validation_split(X_train, y_train, test_size=0.2, random_state=789):
    """
    Split training data into train and validation sets.
    """
    print(f"Creating train/validation split with {test_size} validation ratio...")

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=test_size, random_state=random_state, shuffle=True
    )

    print(f"Split shapes:")
    print(f"X_train_split: {X_train_split.shape}")
    print(f"X_val_split: {X_val_split.shape}")
    print(f"y_train_split: {y_train_split.shape}")
    print(f"y_val_split: {y_val_split.shape}")

    return X_train_split, X_val_split, y_train_split, y_val_split

def scale_features(X_train_split, X_val_split, X_test):
    """
    Apply standard scaling to features.
    """
    print("Applying StandardScaler to features...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val_split)
    X_test_scaled = scaler.transform(X_test)

    print("Feature scaling completed.")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def train_and_evaluate_model(model, X_train_scaled, y_train_split, X_val_scaled, y_val_split, model_name, alpha_value):
    """
    Train a model and evaluate its performance on validation set.
    """
    print(f"Training {model_name} with alpha={alpha_value}...")

    # Train the model
    model.fit(X_train_scaled, y_train_split)

    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train_split, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val_split, y_val_pred))
    train_r2 = r2_score(y_train_split, y_train_pred)
    val_r2 = r2_score(y_val_split, y_val_pred)
    train_mae = mean_absolute_error(y_train_split, y_train_pred)
    val_mae = mean_absolute_error(y_val_split, y_val_pred)

    print(f"{model_name} (alpha={alpha_value}) Performance:")
    print(f"  Train RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
    print(f"  Val   RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}, MAE: {val_mae:.4f}")

    # Count non-zero coefficients for Lasso
    if hasattr(model, 'coef_'):
        if model_name == 'Lasso':
            n_features_selected = np.sum(model.coef_ != 0)
            print(f"  Features selected: {n_features_selected}/{len(model.coef_)}")

    return {
        'model': model,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'alpha': alpha_value
    }

def generate_submission_file(model, X_test_scaled, test_ids, model_name, alpha_value):
    """
    Generate predictions and create submission file.
    """
    print(f"Generating submission for {model_name} with alpha={alpha_value}...")

    # Make predictions on test set
    test_predictions = model.predict(X_test_scaled)

    # Transform predictions back from log space (since we used log1p)
    test_predictions_original = np.expm1(test_predictions)

    # Ensure no negative predictions
    test_predictions_original = np.maximum(test_predictions_original, 0)

    # Create submission DataFrame
    submission = pd.DataFrame({
        'Hospital_Id': test_ids,
        'Transport_Cost': test_predictions_original
    })

    # Create filename
    filename = f'submission_{model_name.lower()}_alpha_{alpha_value}.csv'

    # Save submission
    submission.to_csv(filename, index=False)

    print(f"Submission saved: {filename}")
    print(f"Prediction stats - Mean: {test_predictions_original.mean():.2f}, "
          f"Std: {test_predictions_original.std():.2f}, "
          f"Min: {test_predictions_original.min():.2f}, "
          f"Max: {test_predictions_original.max():.2f}")

    return submission, filename

def comprehensive_model_training():
    """
    Train Ridge and Lasso models with multiple alpha values and generate submissions.
    """
    print("="*80)
    print("COMPREHENSIVE RIDGE AND LASSO MODEL TRAINING")
    print("="*80)

    # Load data
    X_train, y_train, X_test, test_ids = load_processed_datasets()

    # Create validation split
    X_train_split, X_val_split, y_train_split, y_val_split = create_validation_split(
        X_train, y_train, test_size=0.2, random_state=789
    )

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train_split, X_val_split, X_test
    )

    # Define alpha values to test
    alpha_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

    print(f"\nAlpha values to test: {alpha_values}")
    print(f"Total models to train: {len(alpha_values) * 2} (Ridge + Lasso)")

    # Storage for results
    all_results = []
    submission_files = []

    print("\n" + "="*80)
    print("TRAINING RIDGE MODELS")
    print("="*80)

    # Train Ridge models
    for i, alpha in enumerate(alpha_values, 1):
        print(f"\n--- RIDGE MODEL {i}/{len(alpha_values)} ---")

        # Create Ridge model
        ridge_model = Ridge(alpha=alpha, random_state=321, max_iter=10000)

        # Train and evaluate
        ridge_results = train_and_evaluate_model(
            ridge_model, X_train_scaled, y_train_split, X_val_scaled, y_val_split,
            'Ridge', alpha
        )

        # Generate submission
        submission, filename = generate_submission_file(
            ridge_model, X_test_scaled, test_ids, 'Ridge', alpha
        )

        # Store results
        ridge_results['model_type'] = 'Ridge'
        ridge_results['submission_file'] = filename
        all_results.append(ridge_results)
        submission_files.append(filename)

    print("\n" + "="*80)
    print("TRAINING LASSO MODELS")
    print("="*80)

    # Train Lasso models
    for i, alpha in enumerate(alpha_values, 1):
        print(f"\n--- LASSO MODEL {i}/{len(alpha_values)} ---")

        # Create Lasso model
        lasso_model = Lasso(alpha=alpha, random_state=654, max_iter=10000)

        # Train and evaluate
        lasso_results = train_and_evaluate_model(
            lasso_model, X_train_scaled, y_train_split, X_val_scaled, y_val_split,
            'Lasso', alpha
        )

        # Generate submission
        submission, filename = generate_submission_file(
            lasso_model, X_test_scaled, test_ids, 'Lasso', alpha
        )

        # Store results
        lasso_results['model_type'] = 'Lasso'
        lasso_results['submission_file'] = filename
        all_results.append(lasso_results)
        submission_files.append(filename)

    # Create results summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    results_df = pd.DataFrame(all_results)

    print("\nBest models by validation RMSE:")
    best_ridge = results_df[results_df['model_type'] == 'Ridge'].loc[results_df[results_df['model_type'] == 'Ridge']['val_rmse'].idxmin()]
    best_lasso = results_df[results_df['model_type'] == 'Lasso'].loc[results_df[results_df['model_type'] == 'Lasso']['val_rmse'].idxmin()]

    print(f"\nBest Ridge: alpha={best_ridge['alpha']}, Val RMSE={best_ridge['val_rmse']:.4f}, Val R²={best_ridge['val_r2']:.4f}")
    print(f"Best Lasso: alpha={best_lasso['alpha']}, Val RMSE={best_lasso['val_rmse']:.4f}, Val R²={best_lasso['val_r2']:.4f}")

    # Overall best model
    overall_best = results_df.loc[results_df['val_rmse'].idxmin()]
    print(f"\nOverall Best: {overall_best['model_type']} with alpha={overall_best['alpha']}")
    print(f"Val RMSE: {overall_best['val_rmse']:.4f}, Val R²: {overall_best['val_r2']:.4f}")
    print(f"Best submission file: {overall_best['submission_file']}")

    # Save detailed results
    results_df.to_csv('model_comparison_results.csv', index=False)
    print(f"\nDetailed results saved to 'model_comparison_results.csv'")

    print(f"\nAll submission files generated ({len(submission_files)}):")
    for filename in submission_files:
        print(f"  - {filename}")

    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)

    return results_df, submission_files

# Execute the comprehensive training
if __name__ == "__main__":
    results, submissions = comprehensive_model_training()
