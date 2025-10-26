import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from preproc import Preprocessor  # assuming preproc.py defines Preprocessor class

# Load training data
df = pd.read_csv('train.csv')

# Initialize and fit preprocessor on training data
preprocessor = Preprocessor()
preprocessor.fit(df)
df_processed = preprocessor.transform(df)

# Separate features and target
X = df_processed.drop(columns=['Transport_Cost'])
y = df_processed['Transport_Cost']

# Train/validation split (for evaluation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Base estimator: Decision Tree
base_estimator = DecisionTreeRegressor(
    max_depth=300,
    min_samples_split=20,
    max_leaf_nodes=50,
    random_state=42
)

# Initialize AdaBoost Regressor
adaboost_regressor = AdaBoostRegressor(
    estimator=base_estimator,     # use 'estimator' instead of deprecated 'base_estimator'
    n_estimators=100,             # number of weak learners
    learning_rate=0.1,            # weight of each regressor
    random_state=2
)

# Train AdaBoost model
adaboost_regressor.fit(X_train, y_train)

# Evaluate on validation set
y_pred_val = adaboost_regressor.predict(X_val)
mse = mean_squared_error(y_val, y_pred_val)
rmse = mse ** 0.5
r2 = r2_score(y_val, y_pred_val)
print(f"AdaBoost Regression Performance on Validation Set:")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# ---- Predict on test data ----
test_df = pd.read_csv('test.csv')

# Use the first column as ID
id_col = test_df.columns[0]
ids = test_df[id_col]

# Preprocess test data (excluding ID column)
test_features = test_df.drop(columns=[id_col])
test_processed = preprocessor.transform(test_features)

# Predict Transport_Cost
test_predictions = adaboost_regressor.predict(test_processed)

# Create submission dataframe (500 x 2)
submission_df = pd.DataFrame({
    id_col: ids,
    'Transport_Cost': test_predictions
})

# Save to CSV
submission_df.to_csv('submission_adaboost.csv', index=False)
