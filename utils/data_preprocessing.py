import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}

    def fill_missing(self, df: pd.DataFrame, strategy: str = "mean", columns: List[str] = None) -> pd.DataFrame:
        if columns is None:
            columns = df.columns.tolist()
        for col in columns:
            if df[col].isnull().any():
                if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(0)
        return df

    def scale_features(self, df: pd.DataFrame, columns: List[str], method: str = "standard") -> pd.DataFrame:
        if method == "standard":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        self.scalers[method] = scaler
        return df

    def encode_categorical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(df[columns])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns), index=df.index)
        df = df.drop(columns=columns)
        df = pd.concat([df, encoded_df], axis=1)
        self.encoders["onehot"] = encoder
        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        if "co2_emissions_mt" in df.columns and "population_millions" in df.columns:
            df["emissions_per_capita"] = df["co2_emissions_mt"] / (df["population_millions"] * 1e6)
        if "gdp_billions_usd" in df.columns and "population_millions" in df.columns:
            df["gdp_per_capita"] = (df["gdp_billions_usd"] * 1e9) / (df["population_millions"] * 1e6)
        return df
