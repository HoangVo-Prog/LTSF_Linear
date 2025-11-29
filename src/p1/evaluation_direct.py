# evaluation_direct.py

from typing import Tuple
import numpy as np
import pandas as pd

def compute_endpoint_price_from_direct(
    df_segment: pd.DataFrame,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Từ df_segment (đã có lp, y_direct) và y_pred (log-return 100d dự báo)
    trả về:
      - price_true: exp(lp_t + y_direct)
      - price_hat:  exp(lp_t + y_pred)
    """
    lp = df_segment["lp"].values
    y_true = df_segment["y_direct"].values

    # endpoint price
    price_true = np.exp(lp + y_true)
    price_hat = np.exp(lp + y_pred)

    return price_true, price_hat


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def evaluate_model_on_fold_direct_price(
    df_direct: pd.DataFrame,
    model,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    feature_cols,
) -> float:
    """
    Hàm evaluate dùng trong debug hoặc offline,
    nhưng quan trọng là nó dùng chung compute_endpoint_price_from_direct.
    """
    df_train = df_direct.loc[train_mask]
    df_val = df_direct.loc[val_mask]

    X_train = df_train[feature_cols]
    y_train = df_train["y_direct"].values

    X_val = df_val[feature_cols]

    model.fit(X_train, y_train)
    y_pred = model.predict_100day_return(X_val)

    price_true, price_hat = compute_endpoint_price_from_direct(df_val, y_pred)
    return mse(price_true, price_hat)
