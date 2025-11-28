from typing import Tuple
import numpy as np
import pandas as pd

from config import HORIZON


def build_multi_step_target(
    df: pd.DataFrame,
    horizon: int = HORIZON,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Tạo target vector y_multi[t] = [r_{t+1}, ..., r_{t+H}],
    trong đó r_{t+k} = lp_{t+k} - lp_{t+k-1}.
    Trả về:
      df_feat_aligned
      Y_multi: numpy array shape (N, horizon)
    """
    df = df.copy()
    lp = df["lp"].values
    n = len(df)

    Y = []
    valid_indices = []

    for t in range(n):
        if t + horizon < n:
            r_vec = lp[t + 1 : t + horizon + 1] - lp[t : t + horizon]
            Y.append(r_vec)
            valid_indices.append(t)

    Y_array = np.vstack(Y)
    df_out = df.iloc[valid_indices].reset_index(drop=True)
    return df_out, Y_array
