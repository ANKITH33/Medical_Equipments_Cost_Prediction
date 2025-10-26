import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(123)


# ============================================================
# 1. Load Data
# ============================================================

def load_processed_datasets():
    """Load preprocessed datasets from v5 enhanced pipeline."""
    print("Loading preprocessed datasets...")

    X_train = pd.read_csv('X_train_processed_v5_enhanced.csv')
    y_train = pd.read_csv('y_train_processed_v5_enhanced.csv')
    X_test = pd.read_csv('X_test_processed_v5_enhanced.csv')
    test_ids = pd.read_csv('test_ids_v5_enhanced.csv')

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(test_ids, pd.DataFrame):
        test_ids = test_ids.iloc[:, 0]

    print(f" Loaded datasets:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, test_ids: {test_ids.shape}")
    return X_train, y_train, X_test, test_ids


# ============================================================
# 2. Split & Scale
# ============================================================

def create_validation_split(X_train, y_train, test_size=0.2, random_state=789):
    """Split training data into training and validation sets."""
    print("\nCreating train/validation split...")
    return train_test_split(X_train, y_train, test_size=test_size, random_state=random_state, shuffle=True)


def scale_features(X_train_split, X_val_split, X_test):
    """Apply feature scaling."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val_split)
    X_test_scaled = scaler.transform(X_test)
    print(" Feature scaling complete.")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ============================================================
# 3. Model Evaluation
# ============================================================

def evaluate_model(model, X_train_scaled, y_train_split, X_val_scaled, y_val_split):
    """Train and evaluate an AdaBoost model."""
    model.fit(X_train_scaled, y_train_split)

    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train_split, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val_split, y_val_pred))
    val_r2 = r2_score(y_val_split, y_val_pred)
    val_mae = mean_absolute_error(y_val_split, y_val_pred)

    return train_rmse, val_rmse, val_r2, val_mae


# ============================================================
# 4. Comprehensive AdaBoost Training
# ============================================================

def comprehensive_adaboost_training():
    """Train AdaBoost models, find the best, and output submission."""
    print("=" * 80)
    print("ADABOOST REGRESSOR — AUTOMATED TRAINING & SUBMISSION")
    print("=" * 80)

    # 1. Load data
    X_train, y_train, X_test, test_ids = load_processed_datasets()

    # 2. Split
    X_train_split, X_val_split, y_train_split, y_val_split = create_validation_split(X_train, y_train)

    # 3. Scale
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train_split, X_val_split, X_test)

    # 4. Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
        'max_depth': [2, 3, 5, 10, 20, 50]
    }

    all_results = []
    total = len(param_grid['n_estimators']) * len(param_grid['learning_rate']) * len(param_grid['max_depth'])
    print(f"\nTotal combinations to train: {total}\n")

    counter = 1
    for n_est in param_grid['n_estimators']:
        for lr in param_grid['learning_rate']:
            for depth in param_grid['max_depth']:
                print(f"Training model {counter}/{total} → "
                      f"n_estimators={n_est}, learning_rate={lr}, max_depth={depth}")

                base_estimator = DecisionTreeRegressor(max_depth=depth, random_state=42)
                model = AdaBoostRegressor(
                    estimator=base_estimator,
                    n_estimators=n_est,
                    learning_rate=lr,
                    random_state=67
                )

                train_rmse, val_rmse, val_r2, val_mae = evaluate_model(
                    model, X_train_scaled, y_train_split, X_val_scaled, y_val_split
                )

                print(f"→ Val RMSE={val_rmse:.4f}, R²={val_r2:.4f}, MAE={val_mae:.4f}\n")

                all_results.append({
                    'n_estimators': n_est,
                    'learning_rate': lr,
                    'max_depth': depth,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'val_mae': val_mae
                })
                counter += 1

    results_df = pd.DataFrame(all_results)
    best_params = results_df.loc[results_df['val_rmse'].idxmin()]

    print("\n" + "=" * 80)
    print(" BEST ADABOOST MODEL FOUND")
    print("=" * 80)
    print(best_params)

    # 5. Retrain on full dataset
    print("\nRetraining best AdaBoost model on FULL training data...")
    final_base = DecisionTreeRegressor(max_depth=int(best_params['max_depth']), random_state=42)
    final_model = AdaBoostRegressor(
        estimator=final_base,
        n_estimators=int(best_params['n_estimators']),
        learning_rate=float(best_params['learning_rate']),
        random_state=42
    )

    scaler_full = StandardScaler()
    X_train_full_scaled = scaler_full.fit_transform(X_train)
    X_test_full_scaled = scaler_full.transform(X_test)

    final_model.fit(X_train_full_scaled, y_train)

    # 6. Predict on test
    print("\nGenerating predictions on test data...")
    test_predictions_log = final_model.predict(X_test_full_scaled)
    test_predictions = np.expm1(test_predictions_log)
    test_predictions = np.maximum(test_predictions, 0)

    # 7. Save submission
    submission = pd.DataFrame({
        'Hospital_Id': test_ids,
        'Transport_Cost': test_predictions
    })
    submission.to_csv('submission_adaboost.csv', index=False)

    print("\n Final submission saved as 'submission_adaboost.csv'")
    print(f"Prediction stats: mean={test_predictions.mean():.2f}, "
          f"std={test_predictions.std():.2f}, "
          f"min={test_predictions.min():.2f}, "
          f"max={test_predictions.max():.2f}")

    return best_params, submission


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    best_params, submission = comprehensive_adaboost_training()
