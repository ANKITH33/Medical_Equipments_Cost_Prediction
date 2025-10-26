import pandas as pd
import numpy as np

class Preprocessor:
    """
    A class to handle all preprocessing steps.
    It learns parameters from the training data and applies them
    to any dataset (train, test, or new data for inference).
    """
    def __init__(self):
        self.imputation_values = {}
        self.one_hot_columns = []
        self.is_fitted = False

    def fit(self, df_train):
        """
        Learns the necessary parameters (imputation values, etc.)
        from the training data.
        """
        print("Fitting preprocessor on training data...")
        df = df_train.copy()

        # --- Learn imputation values from training data ---
        numerical_cols_to_impute = ['Supplier_Reliability', 'Equipment_Height', 'Equipment_Width', 'Equipment_Weight']
        for col in numerical_cols_to_impute:
            self.imputation_values[col] = df[col].mean()

        cat_cols_to_impute = ['Equipment_Type', 'Transport_Method', 'Rural_Hospital', 'CrossBorder_Shipping',
                              'Urgent_Shipping', 'Installation_Service', 'Fragile_Equipment', 'Hospital_Info']
        for col in cat_cols_to_impute:
            self.imputation_values[col] = df[col].mode()[0]

        self.one_hot_columns = ['Equipment_Type', 'Transport_Method', 'Rural_Hospital']
        self.is_fitted = True
        print("Preprocessor fitting complete.")
        return self

    def transform(self, df_input):
        """
        Applies the learned transformations to a new DataFrame.
        """
        if not self.is_fitted:
            raise RuntimeError("You must fit the preprocessor before transforming data!")

        print("Transforming data...")
        df = df_input.copy()

        # --- REMOVED THE .abs() TRANSFORMATION ON THE TARGET VARIABLE ---
        # The target variable will now be processed outside this class
        # to maintain its original positive/negative values.

        # --- Drop High-Cardinality / ID Columns ---
        cols_to_drop = ['Hospital_Id', 'Supplier_Name', 'Hospital_Location']
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # --- Process Date Columns ---
        df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], format='%m/%d/%y', errors='coerce')
        df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], format='%m/%d/%y', errors='coerce')

        mask = df['Delivery_Date'] < df['Order_Placed_Date']
        df.loc[mask, ['Order_Placed_Date', 'Delivery_Date']] = df.loc[mask, ['Delivery_Date', 'Order_Placed_Date']].values

        df['Delivery_Time_in_Days'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days
        df['order_year'] = df['Order_Placed_Date'].dt.year
        df.drop(columns=['Order_Placed_Date', 'Delivery_Date'], inplace=True)

        # --- Impute All Missing Values using stored values ---
        for col, value in self.imputation_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)

        # --- Map Binary Categorical Features ---
        binary_map_yes_no = {'Yes': 1, 'No': 0}
        binary_cols_yes_no = ['CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service', 'Fragile_Equipment']
        for col in binary_cols_yes_no:
            df[col] = df[col].map(binary_map_yes_no).fillna(self.imputation_values[col])

        df['Hospital_Info'] = df['Hospital_Info'].map({'Wealthy': 1, 'Working Class': 0}).fillna(self.imputation_values['Hospital_Info'])

        # --- One-Hot Encode Multi-Category Features ---
        df = pd.get_dummies(df, columns=self.one_hot_columns, drop_first=True, dtype=float)

        # Final check for any remaining NaNs
        df.dropna(inplace=True)

        print("Transformation complete.")
        return df
