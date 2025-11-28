from typing import Dict, List
import numpy as np
import pandas as pd

from config import HORIZON


def compute_endpoint_price_from_direct(
    df: pd.DataFrame,
    R_hat: np.ndarray,
    start_indices: np.ndarray,
    horizon: int = HORIZON,
) -> Dict[str, np.ndarray]:
    """
    Từ dự báo scalar log return 100 ngày, build endpoint price.
    """
    lp = df["lp"].values
    close = df["close"].values

    price_hat_T = []
    price_true_T = []

    for i, t in enumerate(start_indices):
        lp_t = lp[t]
        lp_T_true = lp[t + horizon]
        price_true_T.append(np.exp(lp_T_true))

        lp_T_hat = lp_t + R_hat[i]
        price_hat_T.append(np.exp(lp_T_hat))

    return {
        "price_hat_T": np.array(price_hat_T),
        "price_true_T": np.array(price_true_T),
    }


def mse_endpoint(price_true: np.ndarray, price_hat: np.ndarray) -> float:
    return float(np.mean((price_hat - price_true) ** 2))


def evaluate_direct_model_on_fold(
    df: pd.DataFrame,
    model,
    fold: Dict,
    feature_cols: List[str],
    horizon: int = HORIZON,
) -> float:
    """
    Evaluate 100 day endpoint MSE trên một fold cho pipeline 1.
    """
    train_mask = fold["train_mask"]
    val_mask = fold["val_mask"]
    val_eval_start_indices = fold["val_eval_start_indices"]

    df_train = df.loc[train_mask]
    df_val = df.loc[val_mask]

    # target đã build trước, nằm trong df["y_direct"]
    X_train = df_train[feature_cols]
    y_train = df_train["y_direct"].values

    model.fit(X_train, y_train)

    # lấy features tại t trong val_eval_start_indices
    X_val_eval = df.loc[val_eval_start_indices, feature_cols]
    R_hat = model.predict_100day_return(X_val_eval)

    price_dict = compute_endpoint_price_from_direct(
        df, R_hat, val_eval_start_indices, horizon=horizon
    )

    score = mse_endpoint(price_dict["price_true_T"], price_dict["price_hat_T"])
    return score
