import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(123)


def load_processed_datasets():
    print("Loading preprocessed datasets")

    X_train = pd.read_csv('X_train_processed.csv')
    y_train = pd.read_csv('y_train_processed.csv')
    X_test = pd.read_csv('X_test_processed.csv')
    test_ids = pd.read_csv('test_ids.csv')


    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(test_ids, pd.DataFrame):
        test_ids = test_ids.iloc[:, 0]

    print(f" Datasets loaded:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, test_ids: {test_ids.shape}")
    return X_train, y_train, X_test, test_ids


def create_validation_split(X_train, y_train, test_size=0.2, random_state=789):
    """Create a train/validation split."""
    print("\nSplitting data into training and validation sets")
    return train_test_split(X_train, y_train, test_size=test_size, random_state=random_state, shuffle=True)


def scale_features(X_train_split, X_val_split, X_test):
    """Apply standard scaling (optional but consistent with other models)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val_split)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling complete.")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def evaluate_model(model, X_train_scaled, y_train_split, X_val_scaled, y_val_split):
    """Train and evaluate a Decision Tree."""
    model.fit(X_train_scaled, y_train_split)

    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train_split, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val_split, y_val_pred))
    val_r2 = r2_score(y_val_split, y_val_pred)
    val_mae = mean_absolute_error(y_val_split, y_val_pred)

    return train_rmse, val_rmse, val_r2, val_mae


def comprehensive_decision_tree_training():
    """Runs Decision Tree training, picks the best model, and outputs final submission."""
    print("=" * 80)
    print("DECISION TREE REGRESSOR — AUTOMATED TRAINING & SUBMISSION")
    print("=" * 80)

    X_train, y_train, X_test, test_ids = load_processed_datasets()

    X_train_split, X_val_split, y_train_split, y_val_split = create_validation_split(X_train, y_train)

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train_split, X_val_split, X_test)

    param_grid = {
        'max_depth': [5, 10, 20, 30, 50, 75, 100, None],
        'min_samples_split': [2, 5, 8, 10, 20],
        'min_samples_leaf': [1, 2, 5, 8, 10]
    }

    all_results = []
    print("\nStarting grid search over Decision Tree hyperparameters\n")

    for depth in param_grid['max_depth']:
        for split in param_grid['min_samples_split']:
            for leaf in param_grid['min_samples_leaf']:
                model = DecisionTreeRegressor(
                    max_depth=depth,
                    min_samples_split=split,
                    min_samples_leaf=leaf,
                    random_state=67
                )
                train_rmse, val_rmse, val_r2, val_mae = evaluate_model(
                    model, X_train_scaled, y_train_split, X_val_scaled, y_val_split
                )

                print(f"Depth={depth}, Split={split}, Leaf={leaf} --> Val RMSE={val_rmse:.4f}, Val R²={val_r2:.4f}")

                all_results.append({
                    'max_depth': depth,
                    'min_samples_split': split,
                    'min_samples_leaf': leaf,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2
                })

    results_df = pd.DataFrame(all_results)
    best_params = results_df.loc[results_df['val_rmse'].idxmin()]
    print("\n" + "=" * 80)
    print("BEST MODEL FOUND ")
    print("=" * 80)
    print(best_params)

    print("\nRetraining best model on FULL training data.")
    final_model = DecisionTreeRegressor(
        max_depth=None if pd.isna(best_params['max_depth']) else int(best_params['max_depth'])
        if best_params['max_depth'] is not None else None,
        min_samples_split=int(best_params['min_samples_split']),
        min_samples_leaf=int(best_params['min_samples_leaf']),
        random_state=42
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

    submission.to_csv('submission_decision_tree.csv', index=False)
    print("\n Final submission saved as 'submission_decision_tree.csv'")
    print(f"Prediction stats: mean={test_predictions.mean():.2f}, std={test_predictions.std():.2f}, "
          f"min={test_predictions.min():.2f}, max={test_predictions.max():.2f}")

    return best_params, submission


if __name__ == "__main__":
    best_params, submission = comprehensive_decision_tree_training()
