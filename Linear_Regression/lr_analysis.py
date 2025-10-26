# linear_regression_analysis_mse_v6.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def comprehensive_linear_regression_analysis_mse():
    """
    Comprehensive analysis using MSE metrics and extensive hyperparameter search
    """

    print("="*80)
    print("LINEAR REGRESSION ANALYSIS - MSE METRICS & EXTENSIVE HYPERPARAMETER SEARCH")
    print("="*80)

    # Load preprocessed data
    print("Loading enhanced preprocessed data...")
    try:
        X_train = pd.read_csv('X_train_processed_v5_enhanced.csv')
        X_test = pd.read_csv('X_test_processed_v5_enhanced.csv')
        y_train = pd.read_csv('y_train_processed_v5_enhanced.csv')

        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]

        print(f"Enhanced preprocessed data loaded successfully:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_train shape: {y_train.shape}")

        print(f"\nTarget variable statistics (log-transformed):")
        print(f"  Mean: {y_train.mean():.4f}")
        print(f"  Std: {y_train.std():.4f}")
        print(f"  Min: {y_train.min():.4f}")
        print(f"  Max: {y_train.max():.4f}")

    except FileNotFoundError as e:
        print(f"Error loading enhanced data: {e}")
        return

    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"\nTrain-Validation Split:")
    print(f"  Training: {X_train_split.shape[0]} samples, {X_train_split.shape[1]} features")
    print(f"  Validation: {X_val_split.shape[0]} samples")

    # Initialize results storage
    results = {}

    # 1. Basic Linear Regression
    print("\n" + "="*60)
    print("1. LINEAR REGRESSION ON ENGINEERED FEATURES")
    print("="*60)

    lr_basic = LinearRegression()
    lr_basic.fit(X_train_split, y_train_split)

    train_pred_lr = lr_basic.predict(X_train_split)
    val_pred_lr = lr_basic.predict(X_val_split)

    train_mse_lr = mean_squared_error(y_train_split, train_pred_lr)
    val_mse_lr = mean_squared_error(y_val_split, val_pred_lr)
    train_r2_lr = r2_score(y_train_split, train_pred_lr)
    val_r2_lr = r2_score(y_val_split, val_pred_lr)

    # Convert back from log space for interpretation
    train_mse_original = mean_squared_error(
        np.expm1(y_train_split), np.expm1(train_pred_lr))
    val_mse_original = mean_squared_error(
        np.expm1(y_val_split), np.expm1(val_pred_lr))

    results['LR_Engineered'] = {
        'train_mse_log': train_mse_lr,
        'val_mse_log': val_mse_lr,
        'train_mse_original': train_mse_original,
        'val_mse_original': val_mse_original,
        'train_r2': train_r2_lr,
        'val_r2': val_r2_lr,
        'overfitting': train_mse_lr - val_mse_lr
    }

    print(f"Training MSE (log-space): {train_mse_lr:.4f}")
    print(f"Validation MSE (log-space): {val_mse_lr:.4f}")
    print(f"Training MSE (original): ${train_mse_original:,.2f}")
    print(f"Validation MSE (original): ${val_mse_original:,.2f}")
    print(f"Training R²: {train_r2_lr:.4f}")
    print(f"Validation R²: {val_r2_lr:.4f}")

    # 2. Ridge with Extensive Grid Search
    print("\n" + "="*60)
    print("2. RIDGE REGRESSION - EXTENSIVE HYPERPARAMETER SEARCH")
    print("="*60)

    # Extensive alpha search for Ridge
    ridge_alphas = np.logspace(-6, 6, 50)  # 50 values from 1e-6 to 1e6
    print(f"Testing {len(ridge_alphas)} alpha values for Ridge...")

    ridge_grid = GridSearchCV(
        Ridge(random_state=42),
        {'alpha': ridge_alphas},
        cv=10,  # Increased CV folds
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    ridge_grid.fit(X_train_split, y_train_split)
    best_ridge = ridge_grid.best_estimator_

    train_pred_ridge = best_ridge.predict(X_train_split)
    val_pred_ridge = best_ridge.predict(X_val_split)

    train_mse_ridge = mean_squared_error(y_train_split, train_pred_ridge)
    val_mse_ridge = mean_squared_error(y_val_split, val_pred_ridge)
    train_r2_ridge = r2_score(y_train_split, train_pred_ridge)
    val_r2_ridge = r2_score(y_val_split, val_pred_ridge)

    # Original scale
    train_mse_ridge_orig = mean_squared_error(
        np.expm1(y_train_split), np.expm1(train_pred_ridge))
    val_mse_ridge_orig = mean_squared_error(
        np.expm1(y_val_split), np.expm1(val_pred_ridge))

    results['Ridge_Extensive'] = {
        'best_alpha': ridge_grid.best_params_['alpha'],
        'best_cv_score': -ridge_grid.best_score_,
        'train_mse_log': train_mse_ridge,
        'val_mse_log': val_mse_ridge,
        'train_mse_original': train_mse_ridge_orig,
        'val_mse_original': val_mse_ridge_orig,
        'train_r2': train_r2_ridge,
        'val_r2': val_r2_ridge,
        'overfitting': train_mse_ridge - val_mse_ridge
    }

    print(f"Best Alpha: {ridge_grid.best_params_['alpha']:.6f}")
    print(f"Best CV MSE: {-ridge_grid.best_score_:.4f}")
    print(f"Training MSE (log): {train_mse_ridge:.4f}")
    print(f"Validation MSE (log): {val_mse_ridge:.4f}")
    print(f"Validation MSE (original): ${val_mse_ridge_orig:,.2f}")
    print(f"Validation R²: {val_r2_ridge:.4f}")

    # 3. Lasso with Extensive Grid Search
    print("\n" + "="*60)
    print("3. LASSO - EXTENSIVE HYPERPARAMETER SEARCH")
    print("="*60)

    # Extensive alpha search for Lasso
    lasso_alphas = np.logspace(-6, 2, 50)  # 50 values from 1e-6 to 1e2
    print(f"Testing {len(lasso_alphas)} alpha values for Lasso...")

    lasso_grid = GridSearchCV(
        Lasso(random_state=42, max_iter=20000, tol=1e-4),
        {'alpha': lasso_alphas},
        cv=10,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    lasso_grid.fit(X_train_split, y_train_split)
    best_lasso = lasso_grid.best_estimator_

    # Feature selection analysis
    selected_features = np.sum(best_lasso.coef_ != 0)
    total_features = len(best_lasso.coef_)

    # Get top features by absolute coefficient
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': best_lasso.coef_,
        'Abs_Coefficient': np.abs(best_lasso.coef_)
    })
    top_features = feature_importance[feature_importance['Abs_Coefficient'] > 0].sort_values(
        'Abs_Coefficient', ascending=False)

    train_pred_lasso = best_lasso.predict(X_train_split)
    val_pred_lasso = best_lasso.predict(X_val_split)

    train_mse_lasso = mean_squared_error(y_train_split, train_pred_lasso)
    val_mse_lasso = mean_squared_error(y_val_split, val_pred_lasso)
    train_r2_lasso = r2_score(y_train_split, train_pred_lasso)
    val_r2_lasso = r2_score(y_val_split, val_pred_lasso)

    # Original scale
    train_mse_lasso_orig = mean_squared_error(
        np.expm1(y_train_split), np.expm1(train_pred_lasso))
    val_mse_lasso_orig = mean_squared_error(
        np.expm1(y_val_split), np.expm1(val_pred_lasso))

    results['Lasso_Extensive'] = {
        'best_alpha': lasso_grid.best_params_['alpha'],
        'best_cv_score': -lasso_grid.best_score_,
        'selected_features': selected_features,
        'total_features': total_features,
        'feature_reduction': (total_features - selected_features) / total_features * 100,
        'train_mse_log': train_mse_lasso,
        'val_mse_log': val_mse_lasso,
        'train_mse_original': train_mse_lasso_orig,
        'val_mse_original': val_mse_lasso_orig,
        'train_r2': train_r2_lasso,
        'val_r2': val_r2_lasso,
        'top_features': top_features.head(15)
    }

    print(f"Best Alpha: {lasso_grid.best_params_['alpha']:.6f}")
    print(f"Best CV MSE: {-lasso_grid.best_score_:.4f}")
    print(f"Features Selected: {selected_features}/{total_features} ({selected_features/total_features*100:.1f}%)")
    print(f"Feature Reduction: {(total_features-selected_features)/total_features*100:.1f}%")
    print(f"Validation MSE (log): {val_mse_lasso:.4f}")
    print(f"Validation MSE (original): ${val_mse_lasso_orig:,.2f}")
    print(f"Validation R²: {val_r2_lasso:.4f}")

    print(f"\nTop 15 Selected Features:")
    print(top_features.head(15)[['Feature', 'Coefficient']].to_string(index=False))

    # 4. ElasticNet with Extensive Grid Search
    print("\n" + "="*60)
    print("4. ELASTICNET - EXTENSIVE HYPERPARAMETER SEARCH")
    print("="*60)

    # Extensive grid search for ElasticNet
    elastic_alphas = np.logspace(-6, 2, 30)  # 30 alpha values
    elastic_l1_ratios = np.linspace(0.1, 0.9, 9)  # 9 l1_ratio values

    print(f"Testing {len(elastic_alphas)} alpha values and {len(elastic_l1_ratios)} l1_ratio values...")
    print(f"Total combinations: {len(elastic_alphas) * len(elastic_l1_ratios)}")

    elastic_grid = GridSearchCV(
        ElasticNet(random_state=42, max_iter=20000, tol=1e-4),
        {
            'alpha': elastic_alphas,
            'l1_ratio': elastic_l1_ratios
        },
        cv=10,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    elastic_grid.fit(X_train_split, y_train_split)
    best_elastic = elastic_grid.best_estimator_

    # Feature selection analysis for ElasticNet
    elastic_selected_features = np.sum(best_elastic.coef_ != 0)

    elastic_feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': best_elastic.coef_,
        'Abs_Coefficient': np.abs(best_elastic.coef_)
    })
    elastic_top_features = elastic_feature_importance[elastic_feature_importance['Abs_Coefficient'] > 0].sort_values(
        'Abs_Coefficient', ascending=False)

    train_pred_elastic = best_elastic.predict(X_train_split)
    val_pred_elastic = best_elastic.predict(X_val_split)

    train_mse_elastic = mean_squared_error(y_train_split, train_pred_elastic)
    val_mse_elastic = mean_squared_error(y_val_split, val_pred_elastic)
    train_r2_elastic = r2_score(y_train_split, train_pred_elastic)
    val_r2_elastic = r2_score(y_val_split, val_pred_elastic)

    # Original scale
    train_mse_elastic_orig = mean_squared_error(
        np.expm1(y_train_split), np.expm1(train_pred_elastic))
    val_mse_elastic_orig = mean_squared_error(
        np.expm1(y_val_split), np.expm1(val_pred_elastic))

    results['ElasticNet_Extensive'] = {
        'best_alpha': elastic_grid.best_params_['alpha'],
        'best_l1_ratio': elastic_grid.best_params_['l1_ratio'],
        'best_cv_score': -elastic_grid.best_score_,
        'selected_features': elastic_selected_features,
        'total_features': total_features,
        'feature_reduction': (total_features - elastic_selected_features) / total_features * 100,
        'train_mse_log': train_mse_elastic,
        'val_mse_log': val_mse_elastic,
        'train_mse_original': train_mse_elastic_orig,
        'val_mse_original': val_mse_elastic_orig,
        'train_r2': train_r2_elastic,
        'val_r2': val_r2_elastic,
        'top_features': elastic_top_features.head(15)
    }

    print(f"Best Alpha: {elastic_grid.best_params_['alpha']:.6f}")
    print(f"Best L1 Ratio: {elastic_grid.best_params_['l1_ratio']:.3f}")
    print(f"Best CV MSE: {-elastic_grid.best_score_:.4f}")
    print(f"Features Selected: {elastic_selected_features}/{total_features} ({elastic_selected_features/total_features*100:.1f}%)")
    print(f"Feature Reduction: {(total_features-elastic_selected_features)/total_features*100:.1f}%")
    print(f"Validation MSE (log): {val_mse_elastic:.4f}")
    print(f"Validation MSE (original): ${val_mse_elastic_orig:,.2f}")
    print(f"Validation R²: {val_r2_elastic:.4f}")

    print(f"\nTop 15 Selected Features (ElasticNet):")
    print(elastic_top_features.head(15)[['Feature', 'Coefficient']].to_string(index=False))

    # 5. Feature Engineering Impact Analysis
    print("\n" + "="*60)
    print("5. IMPACT OF FEATURE ENGINEERING")
    print("="*60)

    # Simple features (remove engineered ones)
    engineered_cols = [col for col in X_train.columns if any(keyword in col.lower()
                      for keyword in ['sine', 'cosine', 'area', 'days_to_delivery',
                                     'order_month_num', 'order_weekday'])]

    simple_features = [col for col in X_train.columns if col not in engineered_cols]

    print(f"Total features: {len(X_train.columns)}")
    print(f"Engineered features: {len(engineered_cols)}")
    print(f"Simple features: {len(simple_features)}")

    if len(simple_features) > 0:
        X_simple_train = X_train_split[simple_features]
        X_simple_val = X_val_split[simple_features]

        lr_simple = LinearRegression()
        lr_simple.fit(X_simple_train, y_train_split)

        val_pred_simple = lr_simple.predict(X_simple_val)
        val_mse_simple = mean_squared_error(y_val_split, val_pred_simple)
        val_r2_simple = r2_score(y_val_split, val_pred_simple)

        print(f"\nWithout Feature Engineering:")
        print(f"  Validation MSE: {val_mse_simple:.4f}")
        print(f"  Validation R²: {val_r2_simple:.4f}")

        print(f"\nWith Feature Engineering:")
        print(f"  Validation MSE: {val_mse_lr:.4f}")
        print(f"  Validation R²: {val_r2_lr:.4f}")

        improvement = val_mse_simple - val_mse_lr
        r2_improvement = val_r2_lr - val_r2_simple

        print(f"\nImprovement from Feature Engineering:")
        print(f"  MSE Reduction: {improvement:.4f} ({improvement/val_mse_simple*100:.1f}%)")
        print(f"  R² Improvement: {r2_improvement:.4f}")

    # 6. Summary
    print("\n" + "="*80)
    print("EXTENSIVE HYPERPARAMETER SEARCH RESULTS SUMMARY")
    print("="*80)

    summary_data = []
    for model_name, result in results.items():
        summary_data.append({
            'Model': model_name,
            'Val_MSE_Log': result['val_mse_log'],
            'Val_R2': result['val_r2'],
            'Features': result.get('selected_features', result.get('features', len(X_train.columns))),
            'Best_Params': {
                'alpha': result.get('best_alpha', 'N/A'),
                'l1_ratio': result.get('best_l1_ratio', 'N/A')
            }
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.round(4))

    return results, summary_df

if __name__ == "__main__":
    results, summary = comprehensive_linear_regression_analysis_mse()
