from typing import Tuple
import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HORIZON


from typing import Tuple, Optional
import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HORIZON


def build_direct_100d_target(
    df: pd.DataFrame,
    horizon: int = HORIZON,
    target_type: str = "endpoint",  # "endpoint" hoặc "weighted_multi"
    weighted_horizons: Tuple[int, ...] = (20, 50, 100),
    weighted_weights: Tuple[float, ...] = (0.2, 0.3, 0.5),
    huberize: bool = False,
    huber_percentiles: Tuple[float, float] = (1.0, 99.0),
    regime_adjust: bool = False,
    regime_vol_col: str = "vol_60",
    regime_threshold: Optional[float] = None,
    regime_threshold_quantile: float = 0.9,
    regime_shrink_factor: float = 0.6,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Tạo target scalar y_direct[t].

    target_type:
      - "endpoint": y_direct[t] = lp_{t+H} - lp_t  (giống code cũ, default)
      - "weighted_multi": y_direct[t] = sum_i w_i * (lp_{t+h_i} - lp_t)

    Các flag:
      - huberize: nếu True thì clip y_direct theo percentile (huber_percentiles)
      - regime_adjust: nếu True thì nếu vol_60 > threshold thì shrink y_direct[t]

    Tất cả biến đổi đều thực hiện trên space log return.
    """

    df = df.copy()
    lp = df["lp"].values.astype(float)
    n = len(df)

    # 1. Tạo base target y_raw
    y_raw = np.full(shape=n, fill_value=np.nan, dtype=float)

    if target_type == "endpoint":
        # Hành vi y như phiên bản cũ
        for t in range(n):
            if t + horizon < n:
                y_raw[t] = lp[t + horizon] - lp[t]

    elif target_type == "weighted_multi":
        horizons_arr = np.array(weighted_horizons, dtype=int)
        weights_arr = np.array(weighted_weights, dtype=float)

        if horizons_arr.shape[0] != weights_arr.shape[0]:
            raise ValueError("weighted_horizons và weighted_weights phải cùng độ dài")

        max_h = int(horizons_arr.max())

        # Chuẩn hóa weight để sum = 1 cho yên tâm
        s = weights_arr.sum()
        if not np.isclose(s, 1.0):
            weights_arr = weights_arr / s

        for t in range(n):
            if t + max_h < n:
                base_lp = lp[t]
                # r_h = lp_{t+h} - lp_t
                returns = lp[t + horizons_arr] - base_lp
                y_raw[t] = float(np.dot(weights_arr, returns))

    else:
        raise ValueError(f"Unknown target_type = {target_type}")

    # Nếu không có điểm nào đủ tương lai thì return luôn
    if np.all(np.isnan(y_raw)):
        df["y_direct"] = y_raw
        df_out = df.loc[[]].reset_index(drop=True)
        y_out = df_out["y_direct"].copy()
        return df_out, y_out

    y = y_raw.copy()

    # 2. Huberized target: clip theo percentile
    if huberize:
        valid = ~np.isnan(y)
        if valid.sum() > 0:
            low_p, high_p = huber_percentiles
            low, high = np.percentile(y[valid], [low_p, high_p])
            y[valid] = np.clip(y[valid], low, high)

    # 3. Regime adjusted target: shrink khi vol cao
    if regime_adjust:
        if regime_vol_col not in df.columns:
            raise KeyError(f"regime_vol_col '{regime_vol_col}' không có trong df")

        vol = df[regime_vol_col].values.astype(float)
        valid = ~np.isnan(y) & ~np.isnan(vol)

        thr = regime_threshold
        if thr is None and valid.sum() > 0:
            # Threshold theo quantile của vol_60 trên period
            thr = float(np.quantile(vol[valid], regime_threshold_quantile))

        if thr is not None:
            high_vol_mask = (vol > thr) & valid
            y[high_vol_mask] = y[high_vol_mask] * regime_shrink_factor

    # 4. Gán vào df và drop NaN như cũ
    df["y_direct"] = y
    mask = ~np.isnan(df["y_direct"])
    df_out = df.loc[mask].reset_index(drop=True)
    y_out = df_out["y_direct"].copy()

    return df_out, y_out
