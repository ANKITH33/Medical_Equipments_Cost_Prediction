import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

def analyze_feature_importance_extensive():
    """
    Analyze feature importance with extensive hyperparameter search
    """

    print("="*70)
    print("FEATURE IMPORTANCE ANALYSIS - EXTENSIVE SEARCH")
    print("="*70)

    # Load enhanced preprocessed data
    X_train = pd.read_csv('X_train_processed.csv')
    y_train = pd.read_csv('y_train_processed.csv')

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    print(f"Enhanced data loaded:")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Samples: {X_train.shape[0]}")

    # Categorize features
    feature_categories = {
        'Cyclical': [col for col in X_train.columns if 'sine' in col or 'cosine' in col],
        'Engineered_Temporal': [col for col in X_train.columns if any(keyword in col.lower()
                               for keyword in ['days_to_delivery', 'order_month', 'order_weekday', 'order_year'])],
        'Engineered_Physical': [col for col in X_train.columns if 'area' in col.lower()],
        'Original_Numerical': [col for col in X_train.columns if any(keyword in col
                              for keyword in ['Base_Transport_Fee', 'Equipment_Value', 'Equipment_Weight', 'Delivery_Duration_Days'])],
        'One_Hot_Encoded': [col for col in X_train.columns if any(keyword in col
                           for keyword in ['Equipment_Type_', 'Transport_Method_', 'Hospital_Info_'])],
        'Binary_Features': [col for col in X_train.columns if any(keyword in col
                           for keyword in ['Rural_Hospital', 'CrossBorder_Shipping', 'Urgent_Shipping',
                                         'Installation_Service', 'Fragile_Equipment'])]
    }

    print(f"\nFeature categorization:")
    for category, features in feature_categories.items():
        print(f"  {category}: {len(features)} features")

    # Split data
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)

    # Extensive Lasso analysis
    print(f"\n" + "="*60)
    print("EXTENSIVE LASSO FEATURE SELECTION ANALYSIS")
    print("="*60)

    lasso_alphas = np.logspace(-6, 2, 30)
    lasso_cv = GridSearchCV(
        Lasso(random_state=42, max_iter=20000),
        {'alpha': lasso_alphas},
        cv=10,
        scoring='neg_mean_squared_error',
        verbose=1
    )

    lasso_cv.fit(X_train_scaled, y_train_split)
    optimal_lasso = lasso_cv.best_estimator_
    optimal_alpha = lasso_cv.best_params_['alpha']

    print(f"Optimal Lasso alpha: {optimal_alpha:.6f}")
    print(f"Best CV MSE: {-lasso_cv.best_score_:.4f}")

    # Feature importance analysis
    lasso_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': optimal_lasso.coef_,
        'Abs_Coefficient': np.abs(optimal_lasso.coef_)
    })

    lasso_selected = lasso_importance[lasso_importance['Abs_Coefficient'] > 0].copy()
    lasso_selected = lasso_selected.sort_values('Abs_Coefficient', ascending=False)

    print(f"\nLasso Feature Selection Results:")
    print(f"  Total features: {len(X_train.columns)}")
    print(f"  Selected features: {len(lasso_selected)}")
    print(f"  Feature reduction: {(len(X_train.columns) - len(lasso_selected))/len(X_train.columns)*100:.1f}%")

    # Extensive ElasticNet analysis
    print(f"\n" + "="*60)
    print("EXTENSIVE ELASTICNET FEATURE SELECTION ANALYSIS")
    print("="*60)

    elastic_alphas = np.logspace(-6, 2, 20)
    elastic_l1_ratios = np.linspace(0.1, 0.9, 9)

    elastic_cv = GridSearchCV(
        ElasticNet(random_state=42, max_iter=20000),
        {
            'alpha': elastic_alphas,
            'l1_ratio': elastic_l1_ratios
        },
        cv=10,
        scoring='neg_mean_squared_error',
        verbose=1
    )

    elastic_cv.fit(X_train_scaled, y_train_split)
    optimal_elastic = elastic_cv.best_estimator_
    optimal_elastic_alpha = elastic_cv.best_params_['alpha']
    optimal_elastic_l1 = elastic_cv.best_params_['l1_ratio']

    print(f"Optimal ElasticNet alpha: {optimal_elastic_alpha:.6f}")
    print(f"Optimal ElasticNet l1_ratio: {optimal_elastic_l1:.3f}")
    print(f"Best CV MSE: {-elastic_cv.best_score_:.4f}")

    # ElasticNet feature importance
    elastic_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': optimal_elastic.coef_,
        'Abs_Coefficient': np.abs(optimal_elastic.coef_)
    })

    elastic_selected = elastic_importance[elastic_importance['Abs_Coefficient'] > 0].copy()
    elastic_selected = elastic_selected.sort_values('Abs_Coefficient', ascending=False)

    print(f"\nElasticNet Feature Selection Results:")
    print(f"  Total features: {len(X_train.columns)}")
    print(f"  Selected features: {len(elastic_selected)}")
    print(f"  Feature reduction: {(len(X_train.columns) - len(elastic_selected))/len(X_train.columns)*100:.1f}%")

    # Compare feature selections
    print(f"\n" + "="*60)
    print("FEATURE SELECTION COMPARISON")
    print("="*60)

    lasso_features = set(lasso_selected['Feature'])
    elastic_features = set(elastic_selected['Feature'])

    common_features = lasso_features & elastic_features
    lasso_only = lasso_features - elastic_features
    elastic_only = elastic_features - lasso_features

    print(f"Features selected by both methods: {len(common_features)}")
    print(f"Features selected only by Lasso: {len(lasso_only)}")
    print(f"Features selected only by ElasticNet: {len(elastic_only)}")

    print(f"\nTop 20 features by Lasso:")
    print(lasso_selected.head(20)[['Feature', 'Coefficient']].to_string(index=False))

    print(f"\nTop 20 features by ElasticNet:")
    print(elastic_selected.head(20)[['Feature', 'Coefficient']].to_string(index=False))

    # Analyze by category
    print(f"\n" + "="*60)
    print("FEATURE SELECTION BY CATEGORY")
    print("="*60)

    for category, cat_features in feature_categories.items():
        if len(cat_features) > 0:
            lasso_in_cat = len([f for f in cat_features if f in lasso_features])
            elastic_in_cat = len([f for f in cat_features if f in elastic_features])

            lasso_pct = lasso_in_cat / len(cat_features) * 100
            elastic_pct = elastic_in_cat / len(cat_features) * 100

            print(f"{category}:")
            print(f"  Total features: {len(cat_features)}")
            print(f"  Lasso selected: {lasso_in_cat} ({lasso_pct:.1f}%)")
            print(f"  ElasticNet selected: {elastic_in_cat} ({elastic_pct:.1f}%)")

    return lasso_selected, elastic_selected, feature_categories, optimal_alpha, optimal_elastic_alpha, optimal_elastic_l1

if __name__ == "__main__":
    lasso_features, elastic_features, categories, lasso_alpha, elastic_alpha, elastic_l1 = analyze_feature_importance_extensive()
