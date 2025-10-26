import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def compare_models_v5():
    """
    Compare different models on enhanced preprocessed data
    """

    print("="*70)
    print("MODEL COMPARISON - ENHANCED PREPROCESSING")
    print("="*70)

    # Load data
    X_train = pd.read_csv('X_train_processed_v5_enhanced.csv')
    y_train = pd.read_csv('y_train_processed_v5_enhanced.csv')

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    print(f"Data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Target: Log-transformed transport cost")

    # Define models
    models = {
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'Ridge (α=0.1)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=0.1, random_state=42))
        ]),
        'Ridge (α=1.0)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0, random_state=42))
        ]),
        'Ridge (α=10.0)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=10.0, random_state=42))
        ]),
        'Lasso (α=0.001)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=0.001, random_state=42, max_iter=10000))
        ]),
        'Lasso (α=0.01)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=0.01, random_state=42, max_iter=10000))
        ]),
        'ElasticNet (α=0.1, l1=0.5)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000))
        ]),
        'Random Forest (baseline)': RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
    }

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print(f"\nPerforming 5-fold cross-validation...")

    for name, model in models.items():
        print(f"  Evaluating {name}...")

        # RMSE scores
        rmse_scores = -cross_val_score(model, X_train, y_train,
                                      cv=cv, scoring='neg_root_mean_squared_error')

        # R² scores
        r2_scores = cross_val_score(model, X_train, y_train,
                                   cv=cv, scoring='r2')

        results[name] = {
            'RMSE_mean': rmse_scores.mean(),
            'RMSE_std': rmse_scores.std(),
            'R2_mean': r2_scores.mean(),
            'R2_std': r2_scores.std()
        }

    # Results summary
    print(f"\n" + "="*80)
    print("CROSS-VALIDATION RESULTS ON ENHANCED FEATURES")
    print("="*80)

    summary_data = []
    for name, result in results.items():
        summary_data.append({
            'Model': name,
            'RMSE_Mean': result['RMSE_mean'],
            'RMSE_Std': result['RMSE_std'],
            'R2_Mean': result['R2_mean'],
            'R2_Std': result['R2_std']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('RMSE_Mean')

    print(summary_df.round(4).to_string(index=False))

    # Best performers
    best_model = summary_df.iloc[0]['Model']
    best_rmse = summary_df.iloc[0]['RMSE_Mean']
    best_r2 = summary_df.iloc[0]['R2_Mean']

    print(f"\nBest performing model: {best_model}")
    print(f"  RMSE: {best_rmse:.4f} ± {summary_df.iloc[0]['RMSE_Std']:.4f}")
    print(f"  R²: {best_r2:.4f} ± {summary_df.iloc[0]['R2_Std']:.4f}")

    # Convert to original scale (approximate)
    print(f"\nApproximate performance in original scale:")
    print(f"  Best RMSE: ~${np.exp(best_rmse):,.0f}")

    return results, summary_df

if __name__ == "__main__":
    results, summary = compare_models_v5()
