import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(123)


def load_processed_datasets():
    """Load the preprocessed datasets from v5 enhanced pipeline."""
    print("Loading preprocessed datasets.")

    X_train = pd.read_csv('X_train_processed.csv')
    y_train = pd.read_csv('y_train_processed.csv')
    X_test = pd.read_csv('X_test_processed.csv')
    test_ids = pd.read_csv('test_ids.csv')

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(test_ids, pd.DataFrame):
        test_ids = test_ids.iloc[:, 0]

    print(f" Loaded datasets:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, test_ids: {test_ids.shape}")

    return X_train, y_train, X_test, test_ids


def create_validation_split(X_train, y_train, test_size=0.2, random_state=789):
    """Create train/validation split."""
    print("\nCreating train/validation split.")
    return train_test_split(X_train, y_train, test_size=test_size, random_state=random_state, shuffle=True)


def scale_features(X_train_split, X_val_split, X_test):
    """Apply scaling to keep consistent with other models."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val_split)
    X_test_scaled = scaler.transform(X_test)
    print(" Feature scaling done.")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def evaluate_model(model, X_train_scaled, y_train_split, X_val_scaled, y_val_split):
    """Train and evaluate a Random Forest model."""
    model.fit(X_train_scaled, y_train_split)
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train_split, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val_split, y_val_pred))
    val_r2 = r2_score(y_val_split, y_val_pred)
    val_mae = mean_absolute_error(y_val_split, y_val_pred)

    return train_rmse, val_rmse, val_r2, val_mae


def comprehensive_random_forest_training():
    """Train Random Forest models, pick the best, and output final submission."""
    print("=" * 80)
    print("RANDOM FOREST REGRESSOR — AUTOMATED TRAINING & SUBMISSION")
    print("=" * 80)

    X_train, y_train, X_test, test_ids = load_processed_datasets()

    X_train_split, X_val_split, y_train_split, y_val_split = create_validation_split(X_train, y_train)

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train_split, X_val_split, X_test)

    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, 30, 50, 100, None],
        'min_samples_split': [2, 5, 8, 10, 20],
        'min_samples_leaf': [1, 2, 4, 5, 10],
        'max_features': ['sqrt', 'log2']
    }

    all_results = []
    total_combinations = (
        len(param_grid['n_estimators']) *
        len(param_grid['max_depth']) *
        len(param_grid['min_samples_split']) *
        len(param_grid['min_samples_leaf']) *
        len(param_grid['max_features'])
    )
    print(f"\nTotal combinations to train: {total_combinations}\n")

    counter = 1
    for n_est in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            for split in param_grid['min_samples_split']:
                for leaf in param_grid['min_samples_leaf']:
                    for feat in param_grid['max_features']:
                        print(f"Training model {counter}/{total_combinations} → "
                              f"n_estimators={n_est}, depth={depth}, split={split}, leaf={leaf}, features={feat}")

                        model = RandomForestRegressor(
                            n_estimators=n_est,
                            max_depth=depth,
                            min_samples_split=split,
                            min_samples_leaf=leaf,
                            max_features=feat,
                            random_state=67,
                            n_jobs=-1
                        )

                        train_rmse, val_rmse, val_r2, val_mae = evaluate_model(
                            model, X_train_scaled, y_train_split, X_val_scaled, y_val_split
                        )

                        all_results.append({
                            'n_estimators': n_est,
                            'max_depth': depth,
                            'min_samples_split': split,
                            'min_samples_leaf': leaf,
                            'max_features': feat,
                            'val_rmse': val_rmse,
                            'val_r2': val_r2,
                            'val_mae': val_mae
                        })

                        print(f"→ Val RMSE={val_rmse:.4f}, R²={val_r2:.4f}, MAE={val_mae:.4f}\n")
                        counter += 1

    results_df = pd.DataFrame(all_results)

    best_params = results_df.loc[results_df['val_rmse'].idxmin()]
    print("\n" + "=" * 80)
    print(" BEST RANDOM FOREST MODEL FOUND")
    print("=" * 80)
    print(best_params)

    print("\nRetraining best Random Forest model on FULL training data.")
    final_model = RandomForestRegressor(
        n_estimators=int(best_params['n_estimators']),
        max_depth=None if pd.isna(best_params['max_depth']) else (
            int(best_params['max_depth']) if best_params['max_depth'] is not None else None
        ),
        min_samples_split=int(best_params['min_samples_split']),
        min_samples_leaf=int(best_params['min_samples_leaf']),
        max_features=best_params['max_features'],
        random_state=42,
        n_jobs=-1
    )

    scaler_full = StandardScaler()
    X_train_full_scaled = scaler_full.fit_transform(X_train)
    X_test_full_scaled = scaler_full.transform(X_test)

    final_model.fit(X_train_full_scaled, y_train)

    print("\nGenerating predictions on test data.")
    test_predictions_log = final_model.predict(X_test_full_scaled)
    test_predictions = np.expm1(test_predictions_log)
    test_predictions = np.maximum(test_predictions, 0)

    submission = pd.DataFrame({
        'Hospital_Id': test_ids,
        'Transport_Cost': test_predictions
    })

    submission.to_csv('submission_random_forest.csv', index=False)
    print("\n Final submission saved as 'submission_random_forest.csv'")
    print(f"Prediction stats: mean={test_predictions.mean():.2f}, "
          f"std={test_predictions.std():.2f}, "
          f"min={test_predictions.min():.2f}, "
          f"max={test_predictions.max():.2f}")

    return best_params, submission

if __name__ == "__main__":
    best_params, submission = comprehensive_random_forest_training()
