import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from preproc import Preprocessor  # assuming preproc.py defines Preprocessor class

# Load data
df = pd.read_csv('train.csv')

# Initialize and fit preprocessor on data
preprocessor = Preprocessor()
preprocessor.fit(df)
df_processed = preprocessor.transform(df)

# Separate features and target
X = df_processed.drop(columns=['Transport_Cost'])
y = df_processed['Transport_Cost']

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Set hyperparameters for Decision Tree
max_depth = 100
min_samples_split = 10
max_leaf_nodes = 20

# Initialize Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    max_leaf_nodes=max_leaf_nodes,
    random_state=2
)

# Train model
dt_regressor.fit(X_train, y_train)

# Predict on validation set
y_pred = dt_regressor.predict(X_val)

# Evaluate performance
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_val, y_pred)

print(f"Decision Tree Regression Performance on Validation Set:")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
