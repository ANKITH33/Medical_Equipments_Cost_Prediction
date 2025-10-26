# 03_gridsearch_models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import the new PreprocessorV2 class
from preproc_v2 import PreprocessorV2

# --- 1. Load Data and Split FIRST ---
print("Loading raw data...")
raw_df = pd.read_csv('train.csv')

# Split the RAW data into training and testing sets
# We pass the full df to the preprocessor to handle target cleaning
train_raw_df, test_raw_df = train_test_split(raw_df, test_size=0.2, random_state=6)

# --- 2. Preprocess Data Correctly ---
preprocessor = PreprocessorV2()

# Fit the preprocessor on the TRAINING data
preprocessor.fit(train_raw_df)

# Transform the training data (with is_training=True to clean the target)
train_processed_df = preprocessor.transform(train_raw_df, is_training=True)
X_train = train_processed_df.drop('Transport_Cost', axis=1)
y_train = train_processed_df['Transport_Cost']

# Transform the test data (with is_training=False)
test_processed_df = preprocessor.transform(test_raw_df, is_training=False)
X_test = test_processed_df.drop('Transport_Cost', axis=1, errors='ignore')
y_test = test_processed_df['Transport_Cost']

# Align columns after one-hot encoding
train_cols = X_train.columns
X_test = X_test.reindex(columns=train_cols, fill_value=0)

# --- 3. Define Combined Pipeline and Grid Search ---
print("\nDefining the combined modeling pipeline and parameter grid...")

# A single pipeline where the model is a parameter to be searched
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(include_bias=False)),
    ('model', Ridge()) # Placeholder model, will be replaced by GridSearchCV
])

# The parameter grid to search over different models and their hyperparameters
param_grid = [
    {
        'poly__degree': [2],
        'model': [Ridge(random_state=32)],
        'model__alpha': np.logspace(2, 5, 4) # [100, 1000, 10000, 100000]
    },
    {
        'poly__degree': [2],
        'model': [Lasso(random_state=12, max_iter=5000)],
        'model__alpha': np.logspace(-2, 2, 5) # [0.01, 0.1, 1, 10, 100]
    },
    {
        'poly__degree': [2],
        'model': [ElasticNet(random_state=24, max_iter=5000)],
        'model__alpha': [0.1, 1.0, 10.0],
        'model__l1_ratio': [0.1, 0.5, 0.9]
    }
]

# --- 4. Run GridSearchCV ---
print("Setting up GridSearchCV to find the best model and parameters...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)

print("Training and searching...")
grid_search.fit(X_train, y_train)

# --- 5. Evaluate the Best Model ---
print("\n--- Grid Search Complete ---")
print(f"Best R² score from cross-validation: {grid_search.best_score_:.4f}")
print("Best parameters found:")
print(grid_search.best_params_)

print("\n--- Evaluating Best Model on the Test Set ---")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Test Set R-squared (R²): {r2:.4f}")
print(f"Test Set Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Test Set Root Mean Squared Error (RMSE): {rmse:,.2f}")
