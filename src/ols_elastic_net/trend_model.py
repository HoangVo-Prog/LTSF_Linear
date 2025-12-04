# trend_model.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from typing import Tuple


class TrendModel:
    def __init__(self, degree: int = 3):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=True)
        self.linear = LinearRegression()

    def fit(self, df: pd.DataFrame, target_col: str = "log_price") -> None:
        t = df["t"].values.reshape(-1, 1)
        X_poly = self.poly.fit_transform(t)
        y = df[target_col].values
        self.linear.fit(X_poly, y)

    def predict_on_index(self, t_index: np.ndarray) -> np.ndarray:
        t_index = np.asarray(t_index).reshape(-1, 1)
        X_poly = self.poly.transform(t_index)
        return self.linear.predict(X_poly)

    def add_trend_and_residual(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        trend = self.predict_on_index(df["t"].values)
        df["trend"] = trend
        df["resid"] = df["log_price"] - df["trend"]
        return df
