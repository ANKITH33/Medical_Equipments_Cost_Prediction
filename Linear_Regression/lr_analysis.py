# linear_regression_analysis_v5.py
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

def comprehensive_linear_regression_analysis_v5():
    """
    Comprehensive analysis using new_preproc_5.py processed data
    """

    print("="*80)
    print("LINEAR REGRESSION ANALYSIS - ENHANCED PREPROCESSING (V5)")
    print("="*80)

    # Load preprocessed data from new_preproc_5.py
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

        # Check if target is log-transformed
        print(f"\nTarget variable statistics (log-transformed):")
        print(f"  Mean: {y_train.mean():.4f}")
        print(f"  Std: {y_train.std():.4f}")
        print(f"  Min: {y_train.min():.4f}")
        print(f"  Max: {y_train.max():.4f}")

        # Show sample of engineered features
        print(f"\nSample of engineered features:")
        feature_samples = X_train.columns[:10].tolist()
        print(f"  First 10 features: {feature_samples}")

        # Check for cyclical features
        cyclical_features = [col for col in X_train.columns if 'sine' in col or 'cosine' in col]
        print(f"  Cyclical features found: {len(cyclical_features)}")
        if cyclical_features:
            print(f"    {cyclical_features}")

        # Check for engineered features
        engineered_features = [col for col in X_train.columns if any(keyword in col.lower()
                              for keyword in ['area', 'delivery', 'days', 'weekday', 'month'])]
        print(f"  Temporal/Physical features: {len(engineered_features)}")
        if engineered_features:
            print(f"    {engineered_features}")

    except FileNotFoundError as e:
        print(f"Error loading enhanced data: {e}")
        print("Please run new_preproc_5_enhanced.py first")
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

    # 1. Basic Linear Regression (on engineered features)
    print("\n" + "="*60)
    print("1. LINEAR REGRESSION ON ENGINEERED FEATURES")
    print("="*60)

    lr_basic = LinearRegression()
    lr_basic.fit(X_train_split, y_train_split)

    train_pred_lr = lr_basic.predict(X_train_split)
    val_pred_lr = lr_basic.predict(X_val_split)

    train_rmse_lr = np.sqrt(mean_squared_error(y_train_split, train_pred_lr))
    val_rmse_lr = np.sqrt(mean_squared_error(y_val_split, val_pred_lr))
    train_r2_lr = r2_score(y_train_split, train_pred_lr)
    val_r2_lr = r2_score(y_val_split, val_pred_lr)

    # Convert back from log space for interpretation
    train_rmse_original = np.sqrt(mean_squared_error(
        np.expm1(y_train_split), np.expm1(train_pred_lr)))
    val_rmse_original = np.sqrt(mean_squared_error(
        np.expm1(y_val_split), np.expm1(val_pred_lr)))

    results['LR_Engineered'] = {
        'train_rmse_log': train_rmse_lr,
        'val_rmse_log': val_rmse_lr,
        'train_rmse_original': train_rmse_original,
        'val_rmse_original': val_rmse_original,
        'train_r2': train_r2_lr,
        'val_r2': val_r2_lr,
        'overfitting': train_rmse_lr - val_rmse_lr
    }

    print(f"Training RMSE (log-space): {train_rmse_lr:.4f}")
    print(f"Validation RMSE (log-space): {val_rmse_lr:.4f}")
    print(f"Training RMSE (original): ${train_rmse_original:,.2f}")
    print(f"Validation RMSE (original): ${val_rmse_original:,.2f}")
    print(f"Training R²: {train_r2_lr:.4f}")
    print(f"Validation R²: {val_r2_lr:.4f}")

    # 2. Ridge with Engineered Features
    print("\n" + "="*60)
    print("2. RIDGE REGRESSION ON ENGINEERED FEATURES")
    print("="*60)

    ridge_alphas = np.logspace(-2, 4, 15)  # Wider range for engineered features
    ridge_grid = GridSearchCV(
        Ridge(random_state=42),
        {'alpha': ridge_alphas},
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    ridge_grid.fit(X_train_split, y_train_split)
    best_ridge = ridge_grid.best_estimator_

    train_pred_ridge = best_ridge.predict(X_train_split)
    val_pred_ridge = best_ridge.predict(X_val_split)

    train_rmse_ridge = np.sqrt(mean_squared_error(y_train_split, train_pred_ridge))
    val_rmse_ridge = np.sqrt(mean_squared_error(y_val_split, val_pred_ridge))
    train_r2_ridge = r2_score(y_train_split, train_pred_ridge)
    val_r2_ridge = r2_score(y_val_split, val_pred_ridge)

    # Original scale
    train_rmse_ridge_orig = np.sqrt(mean_squared_error(
        np.expm1(y_train_split), np.expm1(train_pred_ridge)))
    val_rmse_ridge_orig = np.sqrt(mean_squared_error(
        np.expm1(y_val_split), np.expm1(val_pred_ridge)))

    results['Ridge_Engineered'] = {
        'best_alpha': ridge_grid.best_params_['alpha'],
        'train_rmse_log': train_rmse_ridge,
        'val_rmse_log': val_rmse_ridge,
        'train_rmse_original': train_rmse_ridge_orig,
        'val_rmse_original': val_rmse_ridge_orig,
        'train_r2': train_r2_ridge,
        'val_r2': val_r2_ridge,
        'overfitting': train_rmse_ridge - val_rmse_ridge
    }

    print(f"Best Alpha: {ridge_grid.best_params_['alpha']:.4f}")
    print(f"Training RMSE (log): {train_rmse_ridge:.4f}")
    print(f"Validation RMSE (log): {val_rmse_ridge:.4f}")
    print(f"Training RMSE (original): ${train_rmse_ridge_orig:,.2f}")
    print(f"Validation RMSE (original): ${val_rmse_ridge_orig:,.2f}")
    print(f"Training R²: {train_r2_ridge:.4f}")
    print(f"Validation R²: {val_r2_ridge:.4f}")

    # 3. Lasso with Feature Selection on Engineered Features
    print("\n" + "="*60)
    print("3. LASSO WITH FEATURE SELECTION (ENGINEERED)")
    print("="*60)

    lasso_alphas = np.logspace(-4, 2, 20)  # Fine grid for engineered features
    lasso_grid = GridSearchCV(
        Lasso(random_state=42, max_iter=10000),
        {'alpha': lasso_alphas},
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    lasso_grid.fit(X_train_split, y_train_split)
    best_lasso = lasso_grid.best_estimator_

    # Analyze feature selection
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

    train_rmse_lasso = np.sqrt(mean_squared_error(y_train_split, train_pred_lasso))
    val_rmse_lasso = np.sqrt(mean_squared_error(y_val_split, val_pred_lasso))
    train_r2_lasso = r2_score(y_train_split, train_pred_lasso)
    val_r2_lasso = r2_score(y_val_split, val_pred_lasso)

    # Original scale
    train_rmse_lasso_orig = np.sqrt(mean_squared_error(
        np.expm1(y_train_split), np.expm1(train_pred_lasso)))
    val_rmse_lasso_orig = np.sqrt(mean_squared_error(
        np.expm1(y_val_split), np.expm1(val_pred_lasso)))

    results['Lasso_Engineered'] = {
        'best_alpha': lasso_grid.best_params_['alpha'],
        'selected_features': selected_features,
        'total_features': total_features,
        'feature_reduction': (total_features - selected_features) / total_features * 100,
        'train_rmse_log': train_rmse_lasso,
        'val_rmse_log': val_rmse_lasso,
        'train_rmse_original': train_rmse_lasso_orig,
        'val_rmse_original': val_rmse_lasso_orig,
        'train_r2': train_r2_lasso,
        'val_r2': val_r2_lasso,
        'top_features': top_features.head(10)
    }

    print(f"Best Alpha: {lasso_grid.best_params_['alpha']:.6f}")
    print(f"Features Selected: {selected_features}/{total_features} ({selected_features/total_features*100:.1f}%)")
    print(f"Feature Reduction: {(total_features-selected_features)/total_features*100:.1f}%")
    print(f"Validation RMSE (original): ${val_rmse_lasso_orig:,.2f}")
    print(f"Validation R²: {val_r2_lasso:.4f}")

    print(f"\nTop 10 Selected Features:")
    print(top_features.head(10)[['Feature', 'Coefficient']].to_string(index=False))

    # 4. Polynomial Features on Key Engineered Features
    print("\n" + "="*60)
    print("4. POLYNOMIAL FEATURES ON KEY ENGINEERED FEATURES")
    print("="*60)

    # Select key engineered features for polynomial expansion
    key_features = ['Days_to_Delivery', 'Equipment_Area'] + \
                   [col for col in X_train.columns if 'Base_Transport_Fee' in col or 'Equipment_Value' in col][:3]
    key_features = [f for f in key_features if f in X_train.columns]

    if len(key_features) > 0:
        print(f"Key features for polynomial expansion: {key_features}")

        X_key_train = X_train_split[key_features]
        X_key_val = X_val_split[key_features]

        poly_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('ridge', Ridge(alpha=10.0, random_state=42))
        ])

        poly_pipeline.fit(X_key_train, y_train_split)

        train_pred_poly = poly_pipeline.predict(X_key_train)
        val_pred_poly = poly_pipeline.predict(X_key_val)

        train_rmse_poly = np.sqrt(mean_squared_error(y_train_split, train_pred_poly))
        val_rmse_poly = np.sqrt(mean_squared_error(y_val_split, val_pred_poly))
        train_r2_poly = r2_score(y_train_split, train_pred_poly)
        val_r2_poly = r2_score(y_val_split, val_pred_poly)

        poly_features = poly_pipeline.named_steps['poly'].transform(X_key_train[:1]).shape[1]

        print(f"Original key features: {len(key_features)}")
        print(f"Polynomial features: {poly_features}")
        print(f"Validation RMSE (log): {val_rmse_poly:.4f}")
        print(f"Validation R²: {val_r2_poly:.4f}")

        results['Polynomial_Key'] = {
            'train_rmse_log': train_rmse_poly,
            'val_rmse_log': val_rmse_poly,
            'train_r2': train_r2_poly,
            'val_r2': val_r2_poly,
            'features': poly_features
        }

    # 5. Compare with vs without engineered features
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
        val_rmse_simple = np.sqrt(mean_squared_error(y_val_split, val_pred_simple))
        val_r2_simple = r2_score(y_val_split, val_pred_simple)

        print(f"\nWithout Feature Engineering:")
        print(f"  Validation RMSE: {val_rmse_simple:.4f}")
        print(f"  Validation R²: {val_r2_simple:.4f}")

        print(f"\nWith Feature Engineering:")
        print(f"  Validation RMSE: {val_rmse_lr:.4f}")
        print(f"  Validation R²: {val_r2_lr:.4f}")

        improvement = val_rmse_simple - val_rmse_lr
        r2_improvement = val_r2_lr - val_r2_simple

        print(f"\nImprovement from Feature Engineering:")
        print(f"  RMSE Reduction: {improvement:.4f} ({improvement/val_rmse_simple*100:.1f}%)")
        print(f"  R² Improvement: {r2_improvement:.4f}")

    # Summary
    print("\n" + "="*80)
    print("ENHANCED PREPROCESSING RESULTS SUMMARY")
    print("="*80)

    summary_data = []
    for model_name, result in results.items():
        summary_data.append({
            'Model': model_name,
            'Val_RMSE_Log': result['val_rmse_log'],
            'Val_RMSE_Original': result.get('val_rmse_original', 'N/A'),
            'Val_R2': result['val_r2'],
            'Features': result.get('selected_features', result.get('features', len(X_train.columns)))
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.round(4))

    return results, summary_df

if __name__ == "__main__":
    results, summary = comprehensive_linear_regression_analysis_v5()
