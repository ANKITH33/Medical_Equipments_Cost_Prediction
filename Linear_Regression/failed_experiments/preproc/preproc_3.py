import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    return np, pd, train_test_split


@app.cell
def _():
    INPUT_FILE_PATH = 'train.csv'
    OUTPUT_PROCESSED_PATH = 'processed_data.pkl'
    TARGET_VARIABLE = 'Transport_Cost'
    SEED = 42  # The seed for reproducibility

    return INPUT_FILE_PATH, OUTPUT_PROCESSED_PATH, SEED, TARGET_VARIABLE


@app.cell
def _(np, pd, train_test_split):
    def preprocess_and_save(file_path, target, output_path, random_seed):
        """
        Loads, cleans, preprocesses the data, splits it, and saves to a pickle file.
        """
        print("--- Starting Preprocessing ---")
        df = pd.read_csv(file_path)

        # 1. Cleaning & Feature Engineering
        df[target] = df[target].abs()
        df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], errors='coerce')
        df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')
        df['Delivery_Duration_Days'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days + 1
        df.loc[df['Delivery_Duration_Days'] <= 0, 'Delivery_Duration_Days'] = 1

        # 2. Drop Unused Columns
        cols_to_drop = ['Hospital_Id', 'Supplier_Name', 'Hospital_Location', 'Order_Placed_Date', 'Delivery_Date']
        df = df.drop(columns=cols_to_drop)

        # 3. Imputation
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        numerical_cols.remove(target)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        # 4. Log Transformation
        skewed_features = ['Equipment_Weight', 'Equipment_Value', 'Base_Transport_Fee', 'Delivery_Duration_Days']
        for col in skewed_features:
            df[col] = np.log1p(df[col])
        df[target] = np.log1p(df[target])

        # 5. One-Hot Encoding
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=float)

        # 6. Split Data
        X = df.drop(columns=[target])
        y = df[target]
    
        # Using the random_seed for reproducible split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_seed)

        # 7. Save Processed Data
        processed_data = {'X_train': X_train, 'X_val': X_val, 'y_train': y_train, 'y_val': y_val}
        pd.to_pickle(processed_data, output_path)
    
        print(f"\n✅ Preprocessing complete. Data saved to '{output_path}'")
        print(f"Train set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

    return (preprocess_and_save,)


@app.cell
def _(
    INPUT_FILE_PATH,
    OUTPUT_PROCESSED_PATH,
    SEED,
    TARGET_VARIABLE,
    preprocess_and_save,
):
    preprocess_and_save(INPUT_FILE_PATH, TARGET_VARIABLE, OUTPUT_PROCESSED_PATH, SEED)
    return


if __name__ == "__main__":
    app.run()
