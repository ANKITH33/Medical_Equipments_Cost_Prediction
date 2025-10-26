import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

def train_ridge_lasso_models():
    """
    Train Ridge and Lasso models with validation set
    """

    print("Loading preprocessed data...")
    X_train = pd.read_csv('X_train_preprocessed.csv')
    y_train = pd.read_csv('y_train.csv')
    X_test = pd.read_csv('X_test_preprocessed.csv')

    # Convert y_train to series if it's a DataFrame
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    print(f"Dataset shapes:")
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
        X_train, y_train, test_size=0.15, random_state=42, shuffle=True
    )

    print(f"After split:")
    print(f"X_train_split: {X_train_split.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_train_split: {y_train_split.shape}")
    print(f"y_val: {y_val.shape}")

    # Standardize features
    print(f"\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Features standardized using StandardScaler")

    # Define models
    models = {
        'Ridge': Ridge(alpha=1000.0, random_state=42),
        'Lasso': Lasso(alpha=10.0, random_state=42, max_iter=2000)
    }

    results = {}
    trained_models = {}

    # Train and evaluate models
    print(f"\nTraining models...")
    print("="*60)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train model
        model.fit(X_train_scaled, y_train_split)

        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)

        # Metrics
        train_mse = mean_squared_error(y_train_split, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train_split, y_train_pred)
        train_r2 = r2_score(y_train_split, y_train_pred)

        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = np.sqrt(val_mse)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        # Store results
        results[name] = {
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2
        }

        trained_models[name] = model

        # Print results
        print(f"{name} Results:")
        print(f"  Train - MSE: {train_mse:.2f}, RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.4f}")
        print(f"  Val   - MSE: {val_mse:.2f}, RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}")

    # Select best model based on validation RMSE
    print(f"\n" + "="*60)
    print("MODEL COMPARISON:")
    print("="*60)

    best_model_name = None
    best_val_rmse = float('inf')

    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Validation RMSE: {metrics['val_rmse']:.2f}")
        print(f"  Validation R²: {metrics['val_r2']:.4f}")

        if metrics['val_rmse'] < best_val_rmse:
            best_val_rmse = metrics['val_rmse']
            best_model_name = name

    print(f"\nBest model: {best_model_name} (Validation RMSE: {best_val_rmse:.2f})")

    # Get best model
    best_model = trained_models[best_model_name]

    # Load original test data to get Hospital_Id
    print(f"\nLoading original test data for Hospital_Id...")
    test_original = pd.read_csv('test.csv')

    # Make predictions on test set
    print(f"Making predictions on test set with {best_model_name}...")
    test_predictions = best_model.predict(X_test_scaled)

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
    submission.to_csv('submission.csv', index=False)
    print(f"\nSubmission saved to 'submission.csv'")

    return results, trained_models, best_model_name, submission, scaler

# Run the training
if __name__ == "__main__":
    print("Starting Ridge and Lasso model training...")
    print("="*80)

    results, models, best_model, submission, scaler = train_ridge_lasso_models()

    print("\n" + "="*80)
    print("MODEL TRAINING COMPLETE!")
    print("="*80)

    print(f"\nFinal Results Summary:")
    print(f"Best model: {best_model}")
    print(f"Submission file: submission.csv ({submission.shape[0]} predictions)")

    print(f"\nFiles created:")
    print(f"- submission.csv")

    print("\nTraining completed successfully!")
