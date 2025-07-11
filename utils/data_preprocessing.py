import pandas as pd
from typing import List

class DataPreprocessor:
    """
    Provides methods for preprocessing data, including filling missing values and feature engineering.
    """
    def __init__(self):
        pass

    def fill_missing(self, df: pd.DataFrame, strategy: str = "mean", columns: List[str] = None) -> pd.DataFrame:
        """
        Fill missing values in the specified columns of a DataFrame using the given strategy.

        Args:
            df (pd.DataFrame): The input DataFrame.
            strategy (str): The strategy to use for filling missing values. Options are 'mean', 'median', 'mode', or 'zero'.
            columns (List[str], optional): List of columns to fill. If None, all columns are used.

        Returns:
            pd.DataFrame: DataFrame with missing values filled.
        """
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

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to the DataFrame, such as emissions per capita and GDP per capita, if relevant columns exist.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with new engineered features added.
        """
        if "co2_emissions_mt" in df.columns and "population_millions" in df.columns:
            df["emissions_per_capita"] = df["co2_emissions_mt"] / (df["population_millions"] * 1e6)
        if "gdp_billions_usd" in df.columns and "population_millions" in df.columns:
            df["gdp_per_capita"] = (df["gdp_billions_usd"] * 1e9) / (df["population_millions"] * 1e6)
        return df
