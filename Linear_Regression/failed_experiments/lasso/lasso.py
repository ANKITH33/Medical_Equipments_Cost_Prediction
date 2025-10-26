import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# NEW, CORRECT BLOCK (for Ridge, Lasso, ElasticNet files)
from preprocclass import Preprocessor

# --- 1. Load Data and Split FIRST ---
print("Loading raw data...")
raw_df = pd.read_csv('train.csv')
X_raw = raw_df.drop('Transport_Cost', axis=1)
y_raw = raw_df['Transport_Cost']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# --- 2. Preprocess Data Correctly ---
preprocessor = Preprocessor()
preprocessor.fit(X_train_raw)
X_train = preprocessor.transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# Align columns
train_cols = X_train.columns
test_cols = X_test.columns
missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0
missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0
X_test = X_test[train_cols]

y_train = y_train.loc[X_train.index]
y_test = y_test.loc[X_test.index]

# --- 2. Create the Modeling Pipeline ---
print("Defining the model pipeline...")
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', Lasso(random_state=42, max_iter=2000)) # Increased max_iter for convergence
])

# --- 3. Set up GridSearchCV ---
param_grid = {
    'model__alpha': np.logspace(-3, 7, 14) # [0.1, 1, 10, 100, 1000, 10000]
}

print("Setting up GridSearchCV for Lasso Regression...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

# --- 4. Train the Model ---
print("Training the model and searching for the best alpha...")
grid_search.fit(X_train, y_train)

# --- 5. Evaluate the Best Model ---
print("\n--- Lasso Regression Results ---")
print(f"Best alpha found: {grid_search.best_params_['model__alpha']}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
