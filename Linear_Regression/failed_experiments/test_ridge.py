import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# Import the Preprocessor class we built
from preprocclass import Preprocessor

# --- Configuration ---
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
OUTPUT_FILE = 'submission.csv'
BEST_ALPHA = 100000.0  # The best alpha you found from GridSearchCV

# --- 1. Train the Final Model on ALL Training Data ---
print(f"Loading all training data from '{TRAIN_FILE}' to train the final model...")
train_df_raw = pd.read_csv(TRAIN_FILE)

# Separate features and target from the full training dataset
X_train_full_raw = train_df_raw.drop('Transport_Cost', axis=1)
y_train_full = train_df_raw['Transport_Cost'] # Apply the same transformation to the target

# Create and fit the preprocessor on the FULL training data
preprocessor = Preprocessor()
preprocessor.fit(X_train_full_raw)

# Transform the full training data
X_train_full_processed = preprocessor.transform(X_train_full_raw)

# Define the final model pipeline with the best alpha
print(f"Defining final pipeline with Ridge(alpha={BEST_ALPHA})...")
final_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=BEST_ALPHA, random_state=42))
])

# Train the final pipeline on all processed training data
print("Training the final model on the entire training dataset...")
final_pipeline.fit(X_train_full_processed, y_train_full)
print("Final model training complete.")


# --- 2. Load and Preprocess the Test Data ---
print(f"\nLoading new data for prediction from '{TEST_FILE}'...")
test_df_raw = pd.read_csv(TEST_FILE)

# IMPORTANT: Keep the Hospital_Id for the final submission file
hospital_ids = test_df_raw['Hospital_Id']

# Use the ALREADY FITTED preprocessor to transform the test data
print("Applying learned preprocessing transformations to the test data...")
X_test_processed = preprocessor.transform(test_df_raw)


# --- 3. Align Columns ---
# Ensure the test set has the exact same columns as the training set
print("Aligning columns between training and test sets...")
train_cols = X_train_full_processed.columns
test_cols = X_test_processed.columns

# Add missing columns to the test set (and fill with 0)
missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test_processed[c] = 0

# Ensure the order of columns is the same
X_test_processed = X_test_processed[train_cols]


# --- 4. Make Predictions ---
print("Making predictions on the processed test data...")
predictions = final_pipeline.predict(X_test_processed)

# Ensure predictions are non-negative, as cost cannot be negative
#predictions[predictions < 0] = 0


# --- 5. Create the Submission File ---
print(f"Creating submission file: '{OUTPUT_FILE}'...")
submission_df = pd.DataFrame({
    'Hospital_Id': hospital_ids,
    'Transport_Cost': predictions
})

# Save the DataFrame to a CSV file
submission_df.to_csv(OUTPUT_FILE, index=False)

print("\nPrediction complete!")
print(f"Submission file saved to '{OUTPUT_FILE}'.")
print("\n--- First 5 rows of the submission file: ---")
print(submission_df.head())
