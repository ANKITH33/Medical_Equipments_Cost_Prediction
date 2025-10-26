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

    FILE_PATH = "train.csv"
    return FILE_PATH, np, os, pd, plt, sns


@app.cell
def _(pd):
    def load_and_explore(file_path):
        """
        Loads the dataset and performs a basic initial exploration.
        """
        print(f"Loading data from: {file_path}\n")
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            print("Please make sure the FILE_PATH variable is set correctly.")
            return

        print("--- First 5 Rows ---")
        print(df.head())
        print("\n" + "=" * 50 + "\n")

        print("--- Data Info (Data Types & Non-Null Counts) ---")
        # This will print a concise summary of the DataFrame.
        # It's great for spotting columns with the wrong data type or lots of missing values.
        df.info()
        print("\n" + "=" * 50 + "\n")

        print("--- Descriptive Statistics (for numerical columns) ---")
        # This gives you a statistical summary (mean, std, min, max, etc.)
        print(df.describe())
        print("\n" + "=" * 50 + "\n")

        print("--- Missing Value Counts ---")
        # A direct count of missing values in each column.
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])
        if missing_values.sum() == 0:
            print("No missing values found.")
    return (load_and_explore,)


@app.cell
def _(FILE_PATH, load_and_explore):
    load_and_explore(FILE_PATH)
    return


@app.cell
def _():
    TARGET_VARIABLE = "Transport_Cost"
    return (TARGET_VARIABLE,)


@app.cell
def _(np, os, pd, plt, sns):
    def detailed_eda(file_path, target):
        """
        Performs detailed EDA with visualizations.
        """
        print(f"Loading data from: {file_path}\n")
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return

        # Create a directory to save plots
        if not os.path.exists("plots"):
            os.makedirs("plots")

        print("--- Handling Negative Target Values ---")
        negative_count = (df[target] < 0).sum()
        print(f"Found {negative_count} instances where '{target}' is negative.")
        # For now, we'll treat them as errors and take their absolute value.
        # A more complex approach might be to remove them, but let's see the impact first.
        df[target] = df[target].abs()
        print(f"Converted negative '{target}' values to their absolute values.\n")

        # --- Define Column Types ---
        # High cardinality or identifier columns to drop for now
        cols_to_drop = [
            "Hospital_Id",
            "Supplier_Name",
            "Hospital_Location",
            "Order_Placed_Date",
            "Delivery_Date",
        ]

        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Remove the target variable from the list of numerical features
        if target in numerical_cols:
            numerical_cols.remove(target)

        categorical_cols = [
            col for col in df.columns if col not in numerical_cols + cols_to_drop + [target]
        ]

        print("--- Identified Column Types ---")
        print(f"Numerical Features: {numerical_cols}")
        print(f"Categorical Features: {categorical_cols}")
        print(f"Dropped Features: {cols_to_drop}\n")

        # --- 1. Visualize Numerical Feature Distributions ---
        print("Generating histograms for numerical features...")
        for col in numerical_cols:
            plt.figure(figsize=(10, 5))
            sns.histplot(df[col], kde=True, bins=50)
            plt.title(f"Distribution of {col}")
            plt.savefig(f"plots/hist_{col}.png")
            plt.close()

        # Special visualization for the target variable
        plt.figure(figsize=(10, 5))
        sns.histplot(df[target], kde=True, bins=50)
        plt.title(f"Distribution of Target Variable: {target}")
        plt.savefig(f"plots/hist_TARGET_{target}.png")
        plt.close()

        # And its log-transformed version
        plt.figure(figsize=(10, 5))
        sns.histplot(np.log1p(df[target]), kde=True, bins=50)  # log1p is log(1+x) to handle zeros
        plt.title(f"Distribution of Log-Transformed Target: log({target})")
        plt.savefig(f"plots/hist_TARGET_log_{target}.png")
        plt.close()
        print("...done. Check the 'plots' directory.\n")

        # --- 2. Visualize Categorical Feature Distributions ---
        print("Generating count plots for categorical features...")
        for col in categorical_cols:
            plt.figure(figsize=(12, 6))
            sns.countplot(y=df[col], order=df[col].value_counts().index)
            plt.title(f"Count of {col}")
            plt.tight_layout()
            plt.savefig(f"plots/count_{col}.png")
            plt.close()
        print("...done. Check the 'plots' directory.\n")

        # --- 3. Correlation Heatmap ---
        print("Generating correlation heatmap for numerical features...")
        plt.figure(figsize=(12, 10))
        # Include the target variable in the correlation matrix
        corr_matrix = df[numerical_cols + [target]].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix of Numerical Features")
        plt.tight_layout()
        plt.savefig("plots/correlation_heatmap.png")
        plt.close()
        print("...done. Check the 'plots' directory.\n")

        print("EDA script finished. Please review the generated plots in the 'plots' folder.")
    return (detailed_eda,)


@app.cell
def _(FILE_PATH, TARGET_VARIABLE, detailed_eda):
    detailed_eda(FILE_PATH, TARGET_VARIABLE)
    return


@app.cell
def _(FILE_PATH, pd):
    df = pd.read_csv(FILE_PATH)

    print(f"Original minimum Transport_Cost: {df['Transport_Cost'].min()}")

    # This is the same operation the EDA script performed
    df["Transport_Cost"] = df["Transport_Cost"].abs()

    print(f"Minimum Transport_Cost after .abs(): {df['Transport_Cost'].min()}")
    return


if __name__ == "__main__":
    app.run()
