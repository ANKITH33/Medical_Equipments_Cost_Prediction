import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def extensive_model_comparison():
    """
    Comprehensive model comparison with extensive hyperparameter search and MSE metrics
    """

    print("="*70)
    print("EXTENSIVE MODEL COMPARISON - MSE METRICS")
    print("="*70)

    # Load data
    X_train = pd.read_csv('X_train_processed.csv')
    y_train = pd.read_csv('y_train_processed.csv')

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    print(f"Data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Target: Log-transformed transport cost")

    # Cross-validation setup
    cv = KFold(n_splits=10, shuffle=True, random_state=42)  # Increased to 10-fold

    # Define models with extensive parameter grids
    models_to_test = {}

    # 1. Linear Regression
    models_to_test['Linear Regression'] = {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        'params': {}
    }

    # 2. Ridge with extensive search
    models_to_test['Ridge Regression'] = {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(random_state=42))
        ]),
        'params': {
            'regressor__alpha': np.logspace(-6, 6, 25)
        }
    }

    # 3. Lasso with extensive search
    models_to_test['Lasso Regression'] = {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Lasso(random_state=42, max_iter=20000))
        ]),
        'params': {
            'regressor__alpha': np.logspace(-6, 2, 25)
        }
    }

    # 4. ElasticNet with extensive search
    models_to_test['ElasticNet Regression'] = {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', ElasticNet(random_state=42, max_iter=20000))
        ]),
        'params': {
            'regressor__alpha': np.logspace(-6, 2, 15),
            'regressor__l1_ratio': np.linspace(0.1, 0.9, 9)
        }
    }

    # 5. Random Forest for comparison
    models_to_test['Random Forest'] = {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }

    results = {}
    best_models = {}

    print(f"\nPerforming extensive hyperparameter search with {cv.n_splits}-fold cross-validation...")

    for name, model_config in models_to_test.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {name}...")
        print(f"{'='*50}")

        if model_config['params']:
            # Grid search for hyperparameter optimization
            total_combinations = 1
            for param_values in model_config['params'].values():
                total_combinations *= len(param_values)

            print(f"Testing {total_combinations} parameter combinations...")

            grid_search = GridSearchCV(
                model_config['model'],
                model_config['params'],
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_mse = -grid_search.best_score_

            print(f"Best parameters: {best_params}")
            print(f"Best CV MSE: {best_cv_mse:.4f}")

        else:
            # No hyperparameters to tune
            best_model = model_config['model']
            best_params = {}

            # Evaluate with cross-validation
            mse_scores = -cross_val_score(best_model, X_train, y_train,
                                         cv=cv, scoring='neg_mean_squared_error')
            best_cv_mse = mse_scores.mean()
            mse_std = mse_scores.std()

            print(f"CV MSE: {best_cv_mse:.4f} ± {mse_std:.4f}")

        # Additional evaluation metrics
        r2_scores = cross_val_score(best_model, X_train, y_train,
                                   cv=cv, scoring='r2')

        # For regularized models, check feature selection
        feature_info = {}
        if hasattr(best_model, 'named_steps') and 'regressor' in best_model.named_steps:
            regressor = best_model.named_steps['regressor']
            if hasattr(regressor, 'coef_'):
                non_zero_features = np.sum(regressor.coef_ != 0)
                feature_info = {
                    'selected_features': non_zero_features,
                    'total_features': len(regressor.coef_),
                    'feature_reduction_pct': (len(regressor.coef_) - non_zero_features) / len(regressor.coef_) * 100
                }
        elif hasattr(best_model, 'coef_'):
            non_zero_features = np.sum(best_model.coef_ != 0)
            feature_info = {
                'selected_features': non_zero_features,
                'total_features': len(best_model.coef_),
                'feature_reduction_pct': (len(best_model.coef_) - non_zero_features) / len(best_model.coef_) * 100
            }

        results[name] = {
            'CV_MSE_mean': best_cv_mse,
            'CV_MSE_std': mse_scores.std() if 'mse_scores' in locals() else 0,
            'CV_R2_mean': r2_scores.mean(),
            'CV_R2_std': r2_scores.std(),
            'best_params': best_params,
            **feature_info
        }

        best_models[name] = best_model

        print(f"Final CV R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
        if feature_info:
            print(f"Features selected: {feature_info.get('selected_features', 'N/A')}/{feature_info.get('total_features', 'N/A')}")

    # Results summary
    print(f"\n" + "="*80)
    print("EXTENSIVE MODEL COMPARISON RESULTS")
    print("="*80)

    summary_data = []
    for name, result in results.items():
        summary_data.append({
            'Model': name,
            'CV_MSE_Mean': result['CV_MSE_mean'],
            'CV_MSE_Std': result['CV_MSE_std'],
            'CV_R2_Mean': result['CV_R2_mean'],
            'CV_R2_Std': result['CV_R2_std'],
            'Features_Selected': result.get('selected_features', 'All'),
            'Feature_Reduction_%': result.get('feature_reduction_pct', 0)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('CV_MSE_Mean')

    print(summary_df.round(4).to_string(index=False))

    # Best model analysis
    best_model_name = summary_df.iloc[0]['Model']
    best_mse = summary_df.iloc[0]['CV_MSE_Mean']
    best_r2 = summary_df.iloc[0]['CV_R2_Mean']

    print(f"\n" + "="*60)
    print("BEST MODEL ANALYSIS")
    print("="*60)
    print(f"Best performing model: {best_model_name}")
    print(f"  CV MSE: {best_mse:.4f} ± {summary_df.iloc[0]['CV_MSE_Std']:.4f}")
    print(f"  CV R²: {best_r2:.4f} ± {summary_df.iloc[0]['CV_R2_Std']:.4f}")
    print(f"  Best parameters: {results[best_model_name]['best_params']}")

    # Linear models ranking
    linear_models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet Regression']
    linear_results = summary_df[summary_df['Model'].isin(linear_models)].copy()

    print(f"\n" + "="*60)
    print("LINEAR MODELS RANKING")
    print("="*60)
    print(linear_results[['Model', 'CV_MSE_Mean', 'CV_R2_Mean', 'Features_Selected']].to_string(index=False))

    # Feature selection comparison
    print(f"\n" + "="*60)
    print("FEATURE SELECTION COMPARISON")
    print("="*60)

    for name in ['Lasso Regression', 'ElasticNet Regression']:
        if name in results and 'selected_features' in results[name]:
            result = results[name]
            print(f"{name}:")
            print(f"  Features selected: {result['selected_features']}/{result['total_features']}")
            print(f"  Reduction: {result['feature_reduction_pct']:.1f}%")
            print(f"  Best params: {result['best_params']}")
            print()

    return results, summary_df, best_models

if __name__ == "__main__":
    results, summary, models = extensive_model_comparison()
