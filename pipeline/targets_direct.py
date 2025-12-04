from typing import Tuple
import numpy as np
import pandas as pd
from config import HORIZON


def build_direct_100d_target(
    df: pd.DataFrame,
    horizon: int = HORIZON,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Tạo target scalar y_direct[t] = lp_{t+H} - lp_t.
    Trả về:
      df_feat_aligned: DataFrame sau khi drop các hàng không đủ future.
      y_direct: Series align với df_feat_aligned.index
    """
    df = df.copy()
    lp = df["lp"].values
    n = len(df)

    y = np.full(shape=n, fill_value=np.nan, dtype=float)
    for t in range(n):
        if t + horizon < n:
            y[t] = lp[t + horizon] - lp[t]

    df["y_direct"] = y
    mask = ~np.isnan(df["y_direct"])
    df_out = df.loc[mask].reset_index(drop=True)
    y_out = df_out["y_direct"].copy()

    return df_out, y_out
