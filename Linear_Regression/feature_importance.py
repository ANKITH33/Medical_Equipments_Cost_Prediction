import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def analyze_feature_importance_v5():
    """
    Analyze feature importance using enhanced preprocessed data
    """

    print("="*70)
    print("FEATURE IMPORTANCE ANALYSIS - ENHANCED PREPROCESSING")
    print("="*70)

    # Load enhanced preprocessed data
    X_train = pd.read_csv('X_train_processed_v5_enhanced.csv')
    y_train = pd.read_csv('y_train_processed_v5_enhanced.csv')

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
        if len(features) <= 5:
            print(f"    {features}")
        else:
            print(f"    {features[:3]} ... (+{len(features)-3} more)")

    # Split data
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)

    # Fit Lasso with multiple alphas to see feature selection behavior
    alphas = [0.001, 0.01, 0.1, 1.0]

    for alpha in alphas:
        print(f"\n" + "="*50)
        print(f"LASSO ANALYSIS WITH ALPHA = {alpha}")
        print("="*50)

        lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
        lasso.fit(X_train_scaled, y_train_split)

        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': lasso.coef_,
            'Abs_Coefficient': np.abs(lasso.coef_)
        })

        # Selected features
        selected_features = feature_importance[feature_importance['Abs_Coefficient'] > 0].copy()
        selected_features = selected_features.sort_values('Abs_Coefficient', ascending=False)

        print(f"Features selected: {len(selected_features)}/{len(X_train.columns)} ({len(selected_features)/len(X_train.columns)*100:.1f}%)")

        # Top features by category
        print(f"\nTop 10 selected features:")
        print(selected_features.head(10)[['Feature', 'Coefficient']].to_string(index=False))

        # Analyze by category
        print(f"\nFeature selection by category:")
        for category, cat_features in feature_categories.items():
            selected_in_cat = [f for f in cat_features if f in selected_features['Feature'].values]
            if len(cat_features) > 0:
                percentage = len(selected_in_cat) / len(cat_features) * 100
                print(f"  {category}: {len(selected_in_cat)}/{len(cat_features)} ({percentage:.1f}%)")

                # Show top features in this category
                cat_selected = selected_features[selected_features['Feature'].isin(cat_features)]
                if len(cat_selected) > 0:
                    top_in_cat = cat_selected.head(3)['Feature'].tolist()
                    print(f"    Top: {top_in_cat}")

    # Final analysis with optimal alpha
    print(f"\n" + "="*60)
    print("FINAL FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    # Use cross-validation to find optimal alpha
    from sklearn.model_selection import GridSearchCV

    lasso_cv = GridSearchCV(
        Lasso(random_state=42, max_iter=10000),
        {'alpha': np.logspace(-4, 1, 20)},
        cv=5,
        scoring='neg_root_mean_squared_error'
    )

    lasso_cv.fit(X_train_scaled, y_train_split)
    optimal_alpha = lasso_cv.best_params_['alpha']

    print(f"Optimal alpha from CV: {optimal_alpha:.6f}")

    # Final model with optimal alpha
    final_lasso = lasso_cv.best_estimator_

    final_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': final_lasso.coef_,
        'Abs_Coefficient': np.abs(final_lasso.coef_)
    })

    final_selected = final_importance[final_importance['Abs_Coefficient'] > 0].copy()
    final_selected = final_selected.sort_values('Abs_Coefficient', ascending=False)

    print(f"\nFinal feature selection results:")
    print(f"  Total features: {len(X_train.columns)}")
    print(f"  Selected features: {len(final_selected)}")
    print(f"  Feature reduction: {(len(X_train.columns) - len(final_selected))/len(X_train.columns)*100:.1f}%")

    print(f"\nTop 15 most important features:")
    print(final_selected.head(15)[['Feature', 'Coefficient']].to_string(index=False))

    return final_selected, feature_categories, optimal_alpha

if __name__ == "__main__":
    selected_features, categories, optimal_alpha = analyze_feature_importance_v5()
