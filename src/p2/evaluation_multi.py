# evaluation_multi.py

from typing import Dict, List
import numpy as np
import pandas as pd

from config import HORIZON


def build_price_path_from_returns(
    df: pd.DataFrame,
    start_indices: np.ndarray,
    r_hat_all: np.ndarray,
    horizon: int = HORIZON,
) -> Dict[str, np.ndarray]:
    """
    r_hat_all shape (N_eval, horizon)
    Trả về:
      price_hat_paths: shape (N_eval, horizon)
      price_true_paths: shape (N_eval, horizon)
    """
    lp = df["lp"].values

    n_eval = len(start_indices)
    price_hat_paths = np.zeros((n_eval, horizon))
    price_true_paths = np.zeros((n_eval, horizon))

    for i, t in enumerate(start_indices):
        lp_t = lp[t]

        # ground truth path
        lp_true_segment = lp[t + 1 : t + horizon + 1]
        price_true_segment = np.exp(lp_true_segment)
        price_true_paths[i] = price_true_segment

        # predicted path
        r_vec = r_hat_all[i]
        lp_hat_segment = np.zeros(horizon)
        curr_lp = lp_t
        for k in range(horizon):
            curr_lp = curr_lp + r_vec[k]
            lp_hat_segment[k] = curr_lp

        price_hat_paths[i] = np.exp(lp_hat_segment)

    return {
        "price_hat_paths": price_hat_paths,
        "price_true_paths": price_true_paths,
    }


def mse_full_path(
    price_true_paths: np.ndarray,
    price_hat_paths: np.ndarray,
) -> float:
    diff = price_hat_paths - price_true_paths
    return float(np.mean(diff ** 2))


def evaluate_multi_model_on_fold(
    df: pd.DataFrame,
    Y_multi: np.ndarray,
    model,
    fold: Dict,
    feature_cols: List[str],
    horizon: int = HORIZON,
) -> float:
    """
    Evaluate MSE100 trên một fold cho pipeline 2.

    df:
      - DataFrame có cột lp, time, feature_cols
    Y_multi:
      - numpy array shape (N, H), align theo df.index
    model:
      - MultiStepModel
    """
    train_mask = fold["train_mask"]
    val_mask = fold["val_mask"]
    val_eval_start_indices = fold["val_eval_start_indices"]

    train_idx = df.index[train_mask]

    X_train = df.loc[train_idx, feature_cols]
    Y_train = Y_multi[train_idx]

    model.fit(X_train, Y_train)

    X_val_eval = df.loc[val_eval_start_indices, feature_cols]
    r_hat_all = model.predict_path(X_val_eval)

    price_dict = build_price_path_from_returns(df, val_eval_start_indices, r_hat_all, horizon)
    score = mse_full_path(price_dict["price_true_paths"], price_dict["price_hat_paths"])
    return score
