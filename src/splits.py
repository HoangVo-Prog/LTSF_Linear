from typing import List, Dict
import pandas as pd
import numpy as np

from config import HORIZON


def make_folds(df_time: pd.Series, n_folds: int = 3, horizon: int = HORIZON) -> List[Dict]:
    """
    Sinh các fold time series.
    Ví dụ logic: cut theo mốc ngày cố định hoặc theo tỉ lệ.
    Ở đây chỉ là skeleton, bạn tự chỉnh rule date cụ thể.
    """
    n = len(df_time)
    indices = np.arange(n)

    # Ví dụ chia theo tỉ lệ
    # fold 1: train <= 60 pct, val 60-75 pct
    # fold 2: train <= 75 pct, val 75-90 pct
    # fold 3: train <= 90 pct, val 90-100-horizon pct
    folds = []

    cut_train = [0.60, 0.75, 0.90]
    cut_val = [0.75, 0.90, 1.00]

    for i in range(n_folds):
        train_end = int(cut_train[i] * n)
        val_end = int(cut_val[i] * n)

        train_mask = indices < train_end
        val_mask = (indices >= train_end) & (indices < val_end)

        # Những t trong val sao cho t + horizon < val_end
        valid_t_mask = np.zeros_like(indices, dtype=bool)
        for t in range(n):
            if val_mask[t] and (t + horizon < n) and val_mask[t + horizon]:
                valid_t_mask[t] = True

        val_eval_start_indices = indices[valid_t_mask]

        folds.append(
            {
                "train_mask": train_mask,
                "val_mask": val_mask,
                "val_eval_start_indices": val_eval_start_indices,
            }
        )

    return folds


def get_test_indices(df_time: pd.Series, horizon: int = HORIZON) -> Dict:
    """
    Test set là horizon ngày cuối cùng.
    """
    n = len(df_time)
    test_start = n - horizon
    test_indices = np.arange(test_start, n)
    train_val_mask = np.arange(n) < test_start
    return {
        "test_indices": test_indices,
        "train_val_mask": train_val_mask,
    }
