# residual_model.py

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd                  # add this
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler


class ResidualModel:
    def __init__(self, model_type: str = "elasticnet", use_scaler: bool = True, **kwargs):
        self.model_type = model_type
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None

        if model_type == "elasticnet":
            self._base_model = ElasticNet(**kwargs)
        elif model_type == "ridge":
            self._base_model = Ridge(**kwargs)
        else:
            raise ValueError(f"Unknown residual model_type: {model_type}")

    @property
    def model(self):
        return self._base_model

    def fit(self, X, y):
        if self.use_scaler:
            Xs_arr = self.scaler.fit_transform(X)
            # preserve feature names if X is a DataFrame
            if isinstance(X, pd.DataFrame):
                Xs = pd.DataFrame(Xs_arr, index=X.index, columns=X.columns)
            else:
                Xs = Xs_arr
        else:
            Xs = X
        self._base_model.fit(Xs, y)

    def predict(self, X):
        if self.use_scaler:
            Xs_arr = self.scaler.transform(X)
            if isinstance(X, pd.DataFrame):
                Xs = pd.DataFrame(Xs_arr, index=X.index, columns=X.columns)
            else:
                Xs = Xs_arr
        else:
            Xs = X
        return self._base_model.predict(Xs)

    def score_mse(self, X, y) -> float:
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    def get_coefs(self) -> np.ndarray:
        return getattr(self._base_model, "coef_", None)
