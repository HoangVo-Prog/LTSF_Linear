# models_direct.py

from typing import Any, Dict
import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_STATE

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class Direct100Model:
    """
    Interface chung cho Pipeline 1:
      - fit(X_train, y_train)
      - predict_100day_return(X_input)  -> np.ndarray shape (n,)
    """

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, config: Dict = None) -> None:
        raise NotImplementedError

    def predict_100day_return(self, X_input: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class DirectElasticNetModel(Direct100Model):
    def __init__(self, config: Dict):
        self.config = config or {}
        self.model = ElasticNet(
            alpha=self.config.get("alpha", 1e-3),
            l1_ratio=self.config.get("l1_ratio", 0.5),
            random_state=RANDOM_STATE,
            max_iter=self.config.get("max_iter", 5000),
            fit_intercept=True,
        )

    def fit(self, X_train, y_train, config=None):
        self.model.fit(X_train, y_train)

    def predict_100day_return(self, X_input):
        return self.model.predict(X_input)


class DirectRidgeModel(Direct100Model):
    def __init__(self, config: Dict):
        self.config = config or {}
        self.model = Ridge(
            alpha=self.config.get("alpha", 1.0),
            random_state=RANDOM_STATE,
            fit_intercept=True,
        )

    def fit(self, X_train, y_train, config=None):
        self.model.fit(X_train, y_train)

    def predict_100day_return(self, X_input):
        return self.model.predict(X_input)


class DirectXGBModel(Direct100Model):
    def __init__(self, config: Dict):
        if xgb is None:
            raise ImportError("xgboost not installed")
        self.config = config or {}
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=self.config.get("n_estimators", 500),
            max_depth=self.config.get("max_depth", 4),
            learning_rate=self.config.get("learning_rate", 0.03),
            subsample=self.config.get("subsample", 0.9),
            colsample_bytree=self.config.get("colsample_bytree", 0.9),
            reg_alpha=self.config.get("reg_alpha", 0.0),
            reg_lambda=self.config.get("reg_lambda", 1.0),
            random_state=RANDOM_STATE,
            tree_method=self.config.get("tree_method", "hist"),
            n_jobs=-1,
        )

    def fit(self, X_train, y_train, config=None):
        self.model.fit(X_train, y_train)

    def predict_100day_return(self, X_input):
        return self.model.predict(X_input)


class DirectLGBMModel(Direct100Model):
    def __init__(self, config: Dict):
        if lgb is None:
            raise ImportError("lightgbm not installed")
        self.config = config or {}
        self.model = lgb.LGBMRegressor(
            objective=self.config.get("objective", "regression"),
            metric=self.config.get("metric", "rmse"),
            n_estimators=self.config.get("n_estimators", 500),
            num_leaves=self.config.get("num_leaves", 31),
            max_depth=self.config.get("max_depth", -1),
            learning_rate=self.config.get("learning_rate", 0.03),
            subsample=self.config.get("subsample", 0.9),
            colsample_bytree=self.config.get("colsample_bytree", 0.9),
            min_child_samples=self.config.get("min_child_samples", 20),
            reg_alpha=self.config.get("reg_alpha", 0.0),
            reg_lambda=self.config.get("reg_lambda", 1.0),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,  # tắt spam warning
        )

    def fit(self, X_train, y_train, config=None):
        self.model.fit(X_train, y_train)

    def predict_100day_return(self, X_input):
        return self.model.predict(X_input)


class DirectRandomForestModel(Direct100Model):
    def __init__(self, config: Dict):
        self.config = config or {}
        self.model = RandomForestRegressor(
            n_estimators=self.config.get("n_estimators", 400),
            max_depth=self.config.get("max_depth", None),
            min_samples_leaf=self.config.get("min_samples_leaf", 1),
            max_features=self.config.get("max_features", "sqrt"),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train, config=None):
        self.model.fit(X_train, y_train)

    def predict_100day_return(self, X_input):
        return self.model.predict(X_input)


class DirectGBDTModel(Direct100Model):
    def __init__(self, config: Dict):
        self.config = config
        self.model = GradientBoostingRegressor(
            n_estimators=config.get("n_estimators", 300),
            learning_rate=config.get("learning_rate", 0.05),
            max_depth=config.get("max_depth", 3),
            random_state=RANDOM_STATE,
        )

    def fit(self, X_train, y_train, config=None):
        self.model.fit(X_train, y_train)

    def predict_100day_return(self, X_input):
        return self.model.predict(X_input)


class DirectRollingElasticNetModel(Direct100Model):
    """
    Rolling ElasticNet (placeholder).
    Ý tưởng:
      - fit trên một cửa sổ gần t
      - khi predict t ở validation, model được train trên data trước đó.
    Ở đây để TODO vì logic rolling khá dài.
    """

    def __init__(self, config: Dict):
        self.config = config
        # TODO: lưu window_size, step, v.v.

    def fit(self, X_train, y_train, config=None):
        """
        Có thể không dùng trong CV kiểu classic, mà dùng riêng trong evaluation.
        Ở đây để pass.
        """
        pass

    def predict_100day_return(self, X_input):
        # TODO: implement nếu bạn muốn dùng cùng interface
        raise NotImplementedError("RollingElasticNetModel predict not implemented yet")


class DirectKalmanModel(Direct100Model):
    """
    Placeholder cho Kalman Filter Regression.
    Bạn sẽ cần tự cài thư viện (pykalman, statsmodels, v.v.) rồi implement.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None  # TODO

    def fit(self, X_train, y_train, config=None):
        # TODO: implement Kalman filter regression
        raise NotImplementedError("Kalman regression not implemented yet")

    def predict_100day_return(self, X_input):
        # TODO
        raise NotImplementedError("Kalman regression predict not implemented yet")


class DirectDLinearModel(Direct100Model):
    """
    Placeholder cho DLinear.
    DLinear chuẩn là mô hình theo chuỗi thời gian, cần sequence input.
    Với Pipeline 1 (feature at t -> scalar R_100), bạn có thể implement bản đơn giản.
    """

    def __init__(self, config: Dict):
        self.config = config
        # TODO: torch model

    def fit(self, X_train, y_train, config=None):
        raise NotImplementedError("DLinear not implemented yet")

    def predict_100day_return(self, X_input):
        raise NotImplementedError("DLinear not implemented yet")


class DirectNLinearModel(Direct100Model):
    """
    Placeholder cho NLinear.
    """

    def __init__(self, config: Dict):
        self.config = config

    def fit(self, X_train, y_train, config=None):
        raise NotImplementedError("NLinear not implemented yet")

    def predict_100day_return(self, X_input):
        raise NotImplementedError("NLinear not implemented yet")


# Registry: map tên model -> (class, grid_builder)
MODEL_REGISTRY = {
    "elasticnet": DirectElasticNetModel,
    "ridge": DirectRidgeModel,
    "xgboost": DirectXGBModel,
    "lgbm": DirectLGBMModel,
    "random_forest": DirectRandomForestModel,
    "gbdt": DirectGBDTModel,
    "rolling_elasticnet": DirectRollingElasticNetModel,
    "kalman": DirectKalmanModel,
    "dlinear": DirectDLinearModel,
    "nlinear": DirectNLinearModel,
}
