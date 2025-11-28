# residual_model.py

from typing import Tuple, Dict, Any
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error


class ResidualModel:
    def __init__(self, model_type: str = "elasticnet", **kwargs):
        self.model_type = model_type
        if model_type == "elasticnet":
            self.model = ElasticNet(**kwargs)
        elif model_type == "ridge":
            self.model = Ridge(**kwargs)
        else:
            raise ValueError(f"Unknown residual model_type: {model_type}")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score_mse(self, X, y) -> float:
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)

    def get_coefs(self) -> np.ndarray:
        return getattr(self.model, "coef_", None)
