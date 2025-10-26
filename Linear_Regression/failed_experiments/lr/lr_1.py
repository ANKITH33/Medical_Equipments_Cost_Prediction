import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import the new Preprocessor class
from preprocclass import Preprocessor

# --- 1. Load Data and Split FIRST ---
print("Loading raw data...")
raw_df = pd.read_csv('train.csv')

# Separate raw features and target
X_raw = raw_df.drop('Transport_Cost', axis=1)
y_raw = raw_df['Transport_Cost']

# Split the RAW data into training and testing sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# --- 2. Preprocess Data Correctly ---
# Create an instance of our preprocessor
preprocessor = Preprocessor()

# Fit the preprocessor on the TRAINING data ONLY
preprocessor.fit(X_train_raw)

# Transform both the training and testing data
X_train = preprocessor.transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# Align columns after one-hot encoding to ensure train and test sets have the same features
train_cols = X_train.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0

X_test = X_test[train_cols] # Ensure order is the same

# We also need to process the target variable (y)
y_train = y_train.loc[X_train.index].abs()
y_test = y_test.loc[X_test.index].abs()


# --- 3. Create and Train the Modeling Pipeline ---
print("\nDefining the model pipeline...")
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

print("Training the Linear Regression model...")
pipeline.fit(X_train, y_train)

# --- 4. Evaluate the Model ---
print("\nEvaluating the model on the test set...")
y_pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"--- Linear Regression Results ---")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
