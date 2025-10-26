import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# --- Configuration ---
TRAIN_FILE = 'train.csv'
OUTPUT_DIR = 'eda_numerical_plots'
TARGET_COL = 'Transport_Cost'

# --- Main Analysis Function ---
def analyze_numerical_features():
    """
    Loads, preprocesses, and plots numerical features against the target
    in four different ways: scaled/unscaled vs linear/log target.
    """
    # --- 1. Load and Preprocess Data ---
    print(f"Loading data from '{TRAIN_FILE}'...")
    try:
        df = pd.read_csv(TRAIN_FILE)
    except FileNotFoundError:
        print(f"Error: '{TRAIN_FILE}' not found. Please place it in the same directory.")
        return

    # Apply essential cleaning rules
    df = df[df[TARGET_COL] >= 0].copy()

    # Handle date swapping
    df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
    df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], format='%m/%d/%y', errors='coerce')
    swap_mask = df['Delivery_Date'] < df['Order_Placed_Date']
    df.loc[swap_mask, ['Order_Placed_Date', 'Delivery_Date']] = \
        df.loc[swap_mask, ['Delivery_Date', 'Order_Placed_Date']].values
    df['Delivery_Time_in_Days'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days

    # Identify final numerical features for plotting
    numerical_features = [
        'Supplier_Reliability', 'Equipment_Height', 'Equipment_Width',
        'Equipment_Weight', 'Equipment_Value', 'Base_Transport_Fee',
        'Delivery_Time_in_Days'
    ]

    # Impute missing values with the median for plotting purposes
    for col in numerical_features:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    print("Preprocessing complete.")

    # --- 2. Create Transformed Features and Target ---

    # Create log-transformed target
    # Using log1p to safely handle cases where Transport_Cost might be 0
    df['log_Transport_Cost'] = np.log1p(df[TARGET_COL])

    # Create scaled versions of numerical features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[numerical_features])
    df_scaled = pd.DataFrame(scaled_features, columns=[f"{col}_scaled" for col in numerical_features], index=df.index)

    # Combine into one master DataFrame for easy plotting
    df_analysis = pd.concat([df, df_scaled], axis=1)

    print("Created log-transformed target and scaled features.")

    # --- 3. Generate and Save Plots ---

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: '{OUTPUT_DIR}'")

    print("Generating plots for each numerical feature...")
    for feature in numerical_features:
        scaled_feature = f"{feature}_scaled"

        # Create a 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Analysis of "{feature}" vs. Target', fontsize=16, y=1.02)

        # Plot 1: Unscaled Feature vs. Linear Target
        sns.scatterplot(data=df_analysis, x=feature, y=TARGET_COL, ax=axes[0, 0], alpha=0.4)
        axes[0, 0].set_title(f'Unscaled Feature vs. {TARGET_COL}')

        # Plot 2: Unscaled Feature vs. Log Target
        sns.scatterplot(data=df_analysis, x=feature, y='log_Transport_Cost', ax=axes[0, 1], alpha=0.4)
        axes[0, 1].set_title(f'Unscaled Feature vs. log({TARGET_COL})')

        # Plot 3: Scaled Feature vs. Linear Target
        sns.scatterplot(data=df_analysis, x=scaled_feature, y=TARGET_COL, ax=axes[1, 0], alpha=0.4)
        axes[1, 0].set_title(f'Scaled Feature vs. {TARGET_COL}')

        # Plot 4: Scaled Feature vs. Log Target
        sns.scatterplot(data=df_analysis, x=scaled_feature, y='log_Transport_Cost', ax=axes[1, 1], alpha=0.4)
        axes[1, 1].set_title(f'Scaled Feature vs. log({TARGET_COL})')

        plt.tight_layout()

        # Save the figure
        plot_filename = os.path.join(OUTPUT_DIR, f'{feature}_analysis.png')
        plt.savefig(plot_filename)
        plt.close(fig) # Close the figure to free up memory

    print(f"\nAll plots have been saved to the '{OUTPUT_DIR}' directory.")

# --- Run the analysis ---
if __name__ == '__main__':
    analyze_numerical_features()
