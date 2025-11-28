from typing import Any
import numpy as np
import pandas as pd


class MultiStepModel:
    """
    Interface chung cho pipeline 2, multi output.
    """

    def __init__(self, base_model: Any):
        """
        base_model có thể là:
          - MultiOutputRegressor(ElasticNet)
          - custom model dự báo vector
        """
        self.base_model = base_model

    def fit(self, X_train: pd.DataFrame, Y_train: np.ndarray, config: dict = None) -> None:
        """
        Y_train shape (N, H).
        """
        # TODO: apply config vào base_model nếu cần
        self.base_model.fit(X_train, Y_train)

    def predict_path(self, X_input: pd.DataFrame) -> np.ndarray:
        """
        X_input: DataFrame n_sample x n_feat
        Output: numpy array shape (n_sample, H)
        """
        Y_hat = self.base_model.predict(X_input)
        return np.asarray(Y_hat, dtype=float)
