import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import warnings

    # We can safely ignore the date parsing warning now because we are fixing it.
    warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually.*")

    return np, os, pd, plt, sns


@app.cell
def _():
    TRAIN_FILE_PATH = 'train.csv'
    TARGET_VARIABLE = 'Transport_Cost'
    PLOT_DIR = 'plots/final_feature_distributions'

    return PLOT_DIR, TARGET_VARIABLE, TRAIN_FILE_PATH


@app.cell
def _(PLOT_DIR, TARGET_VARIABLE, TRAIN_FILE_PATH, np, os, pd, plt, sns):
    def analyze_features():
        """
        Runs the full preprocessing pipeline and generates a histogram for every
        single feature that would be fed into the model.
        """
        print("--- Starting Full Feature Analysis ---")
    
        # Create a directory to save the plots
        if not os.path.exists(PLOT_DIR):
            os.makedirs(PLOT_DIR)
        print(f"Plots will be saved in the '{PLOT_DIR}' directory.")

        # 1. Load Data
        df = pd.read_csv(TRAIN_FILE_PATH)

        # --- Start of Full Preprocessing Pipeline ---
    
        # Step A: Initial Cleaning & Date Features (with format fix)
        print("Step A: Cleaning and parsing dates with explicit format...")
        df[TARGET_VARIABLE] = df[TARGET_VARIABLE].abs()
        # FIX: Provide the correct format to pd.to_datetime
        df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
        df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], format='%m/%d/%y', errors='coerce')
        df['Delivery_Duration_Days'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days + 1
        df.loc[df['Delivery_Duration_Days'] <= 0, 'Delivery_Duration_Days'] = 1
    
        # Step B: Imputation
        print("Step B: Imputing missing values...")
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        numerical_cols.remove(TARGET_VARIABLE)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in numerical_cols: df[col] = df[col].fillna(df[col].median())
        for col in categorical_cols: df[col] = df[col].fillna(df[col].mode()[0])

        # Step C: Create new, smart features
        print("Step C: Engineering new features (Volume, Density, etc.)...")
        df['Equipment_Height'] = df['Equipment_Height'].replace(0, 1)
        df['Equipment_Width'] = df['Equipment_Width'].replace(0, 1)
        df['Equipment_Volume'] = df['Equipment_Height'] * df['Equipment_Width']
        df['Value_Density'] = df['Equipment_Value'] / df['Equipment_Volume']
        df['Weight_Density'] = df['Equipment_Weight'] / df['Equipment_Volume']
        df['Service_Level_Score'] = (df['Urgent_Shipping'] == 'Yes').astype(int) + \
                                    (df['Installation_Service'] == 'Yes').astype(int) + \
                                    (df['CrossBorder_Shipping'] == 'Yes').astype(int)
        df.replace([np.inf, -np.inf], 0, inplace=True)

        # Step D: Drop original and unused columns
        cols_to_drop = ['Hospital_Id', 'Supplier_Name', 'Hospital_Location', 'Order_Placed_Date', 'Delivery_Date']
        df = df.drop(columns=cols_to_drop)

        # Step E: One-Hot Encoding
        print("Step E: One-hot encoding categorical features...")
        final_categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        df = pd.get_dummies(df, columns=final_categorical_cols, drop_first=True, dtype=float)

        # Step F: Log Transformation
        print("Step F: Applying log transformations...")
        skewed_features = ['Equipment_Weight', 'Equipment_Value', 'Base_Transport_Fee', 
                           'Delivery_Duration_Days', 'Equipment_Volume', 'Value_Density', 'Weight_Density']
        for col in skewed_features:
            if col in df.columns: df[col] = np.log1p(df[col].astype(float))
        df[TARGET_VARIABLE] = np.log1p(df[TARGET_VARIABLE])
    
        # --- End of Pipeline ---
    
        # 3. Generate a plot for EVERY feature
        print("\nStep G: Generating distribution plots for all final features...")
        X = df.drop(columns=[TARGET_VARIABLE])
    
        for i, col in enumerate(X.columns):
            plt.figure(figsize=(10, 6))
            sns.histplot(X[col], kde=True, bins=30)
        
            # Clean up column names for file saving
            safe_col_name = col.replace('/', '_').replace(' ', '_')
            plt.title(f'Distribution of Final Feature: {col}', fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f'{i:03d}_{safe_col_name}.png'))
            plt.close()

        print(f"\n✅ Analysis complete. {len(X.columns)} plots have been saved to '{PLOT_DIR}'.")
        print("Please review the plots to understand the final data distributions.")

    return (analyze_features,)


@app.cell
def _(analyze_features):
    analyze_features()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
