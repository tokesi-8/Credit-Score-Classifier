from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler


class DataPreprocessing:

    DROP_COLS = ["Unnamed: 0", "ID", "Customer_ID", "SSN", "Name", "Month", "Occupation"]
    NUMERIC_LIKE_OBJECT_COLS = ["Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment","Changed_Credit_Limit", "Amount_invested_monthly","Outstanding_Debt", "Monthly_Balance",]
    NEGATIVE_FIX_COLS = ["Age", "Num_Bank_Accounts", "Num_of_Loan", "Delay_from_due_date","Num_of_Delayed_Payment", "Changed_Credit_Limit", "Monthly_Balance",]
    TARGET_COL = "Credit_Score"
    TARGET_MAPPING = {"Poor": 0, "Standard": 1, "Good": 2}
    
    IQR_CLIP_COLS = ["Age", "Num_Credit_Card", "Num_Bank_Accounts", "Num_of_Loan","Num_Credit_Inquiries", "Num_of_Delayed_Payment", "Annual_Income",]

	# Numeric Cols & Categorical Cols di assign pada tahap setelah fill missing value
    numeric_cols_: list[str] = []
    categorical_cols_: list[str] = []
    
    ORDINAL_COLS = ["Credit_Mix", "Payment_of_Min_Amount"]
    ORDINAL_ORDER = [
        ["Unknown", "Bad", "Standard", "Good"],
        ["Unknown", "No", "Yes"],
    ]
    NOMINAL_COLS = ["Payment_Behaviour"]

	# Initialisasi
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    @staticmethod
    def _clean_numeric_column(series: pd.Series) -> pd.Series:
    # Handling variable numerik yang noisy dan ubah dari string menjadi float
        return (series.astype(str).str.replace(r"[^0-9.-]", "", regex=True).replace("", np.nan).astype(float))

    @staticmethod
    def _convert_credit_history_to_months(value):
    #Handling variable "Credit_History_Age" dari String menjadi Integer dalam month
        if pd.isna(value):
            return np.nan

        years = 0
        months = 0

        if "Year" in value:
            years = int(value.split("Year")[0].strip())
        if "Month" in value:
            months = int(value.split("and")[1].split("Month")[0].strip())

        return years * 12 + months

	# 1. Cleaning & Feature Engineering (dilakukan sebelum split)
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        
        print("--- Step 2.1: Cleaning & Feature Engineering ---")
        
        df = df.copy()

        # Drop Kolom Identifier
        df = df.drop(columns= self.DROP_COLS, errors="ignore")

      	# Pembersihan dan Konversi beberapa Data Numerik
        for col in self.NUMERIC_LIKE_OBJECT_COLS:
            df[col] = self._clean_numeric_column(df[col])

        # Replace value uninformatif pada kategorikal dengan Unknown
        if "Credit_Mix" in df.columns:
            df["Credit_Mix"] = df["Credit_Mix"].replace("_", "Unknown")
        if "Payment_of_Min_Amount" in df.columns:
            df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", "Unknown")
        if "Payment_Behaviour" in df.columns:
            df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", "Unknown")

        # Handling `Type_of_Loan` menjadi `Num_Type_of_Loan`
        if "Type_of_Loan" in df.columns:
            df["Type_of_Loan"] = df["Type_of_Loan"].replace("Not Specified", np.nan)
            df["Num_Type_of_Loan"] = df["Type_of_Loan"].apply(lambda x: len(str(x).split(",")) if pd.notna(x) else 0)
            df = df.drop(columns=["Type_of_Loan"])

        # Handling `Credit_History_Age` menjadi `Credit_History_Age_Months`
        if "Credit_History_Age" in df.columns:
            df["Credit_History_Age_Months"] = df["Credit_History_Age"].apply(self._convert_credit_history_to_months)
            df = df.drop(columns=["Credit_History_Age"])

        # Handling Negative Value
        for col in self.NEGATIVE_FIX_COLS:
            if col in df.columns:
                df[col] = df[col].abs()

        print(f"✅ Cleaning selesai. Shape data: {df.shape}\n\n\n")
        return df

    # 2. Splitting
    def split(self, df: pd.DataFrame):
        
        print("--- Step 2.2: Train-Test Split ---")

        X = df.drop(columns=[self.TARGET_COL])
        y = df[self.TARGET_COL].map(self.TARGET_MAPPING)

        X_train, X_test, y_train, y_test = train_test_split(
    		X, y, test_size=self.test_size, stratify=y, random_state=self.random_state,
        )
        
        print(f"✅ Split selesai. X_train: {X_train.shape}, X_test: {X_test.shape}\n\n\n")
        return X_train, X_test, y_train, y_test

    # 3. Outlier handling (Handling dilakukan based data X_train)
    def handle_outliers(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series):
        
        print("--- Step 2.3: Outlier Handling ---")
        X_train = X_train.copy()
        X_test = X_test.copy()

        # Handling `Monthly_Balance` dengan Drop baris corrupt (> 1e6)
        if "Monthly_Balance" in X_train.columns:
            corrupt_mask_train = X_train["Monthly_Balance"] > 1e6
            n_corrupt = int(corrupt_mask_train.sum())
            if n_corrupt:
                X_train = X_train[~corrupt_mask_train]
                y_train = y_train[~corrupt_mask_train]

        # IQR clipping `Age`, `count variables` dan `Annual_Income`
        for col in self.IQR_CLIP_COLS:
            if col not in X_train.columns:
                continue
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = max(0, Q1 - 1.5 * IQR)
            upper_bound = Q3 + 1.5 * IQR

            X_train[col] = X_train[col].clip(lower=lower_bound, upper=upper_bound)
            X_test[col] = X_test[col].clip(lower=lower_bound, upper=upper_bound)

        # Handling `Monthly_Balance` dengan Percentile capping P99 
        if "Interest_Rate" in X_train.columns:
            upper_bound = X_train["Interest_Rate"].quantile(0.99)
            X_train["Interest_Rate"] = X_train["Interest_Rate"].clip(upper=upper_bound)
            X_test["Interest_Rate"] = X_test["Interest_Rate"].clip(upper=upper_bound)
            
        print("✅ Outlier handling selesai\n\n\n")
        return X_train, X_test, y_train

	
    # 4. Missing value imputation (based on X_train)
    def handle_missing_values(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        print("--- Step 2.4: Missing Value Imputation ---")
        X_train = X_train.copy()
        X_test = X_test.copy()
 
        # Kolom numerik & kategorikal disimpan untuk ColumnTransformer 
        self.numeric_cols_ = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.categorical_cols_ = X_train.select_dtypes(include="object").columns.tolist()
 
        # Median & mode dihitung dari X_train
        train_medians = X_train[self.numeric_cols_].median()
        train_modes = X_train[self.categorical_cols_].mode().iloc[0]
 
        X_train[self.numeric_cols_] = X_train[self.numeric_cols_].fillna(train_medians)
        X_train[self.categorical_cols_] = X_train[self.categorical_cols_].fillna(train_modes)
 
        X_test[self.numeric_cols_] = X_test[self.numeric_cols_].fillna(train_medians)
        X_test[self.categorical_cols_] = X_test[self.categorical_cols_].fillna(train_modes)
 
        print("✅ Missing value imputation selesai\n\n\n")
        return X_train, X_test

 
    # 5. ColumnTransformer untuk encoding & scaling
    def build_preprocessor(self) -> ColumnTransformer:
        print("--- Step 2.5: Build ColumnTransformer (Encoding & Scaling) ---")
 
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])
 
        ordinal_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=self.ORDINAL_ORDER,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])
 
        nominal_transformer = Pipeline(steps=[
            ("encoder", OneHotEncoder(
            	handle_unknown="ignore",
				sparse_output=False
            )),
        ])
 
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, self.numeric_cols_),
            ("ord", ordinal_transformer, self.ORDINAL_COLS),
            ("nom", nominal_transformer, self.NOMINAL_COLS),
        ])
 
        print("✅ ColumnTransformer berhasil dibuat\n\n\n")
        return preprocessor
 
    # Orkestrasi seluruh step preprocessing
    def run(self, df: pd.DataFrame):        
        df_clean = self.clean(df)
        X_train, X_test, y_train, y_test = self.split(df_clean)
        X_train, X_test, y_train = self.handle_outliers(X_train, X_test, y_train)
        X_train, X_test = self.handle_missing_values(X_train, X_test)
        preprocessor = self.build_preprocessor()
 
        return X_train, X_test, y_train, y_test, preprocessor
 