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
    target_type: str = "endpoint",      # "endpoint" hoặc "weighted_multi"
    weighted_horizons=(20, 50, 100),
    weighted_weights=(0.5, 0.3, 0.2),

    huberize: bool = False,
    huber_percentiles=(2.0, 98.0),

    # ----- Regime adjust: volatility -----
    regime_vol_adjust: bool = False,
    regime_vol_col: str = "vol_60",
    regime_vol_quantile: float = 0.9,   # q90 = vol spike
    regime_vol_shrink: float = 0.6,     # shrink 40 percent khi vol cao

    # ----- Regime adjust: drawdown -----
    regime_dd_adjust: bool = False,
    regime_dd_price_col: str = "close", # nếu không có, sẽ dùng exp(lp)
    regime_dd_threshold: float = 0.15,  # drawdown > 15 percent coi là downtrend
    regime_dd_shrink: float = 0.5,      # shrink thêm 50 percent khi DD lớn
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Tạo target scalar y_direct[t] dùng cho direct model.

    target_type:
      - "endpoint": y[t] = lp_{t+H} - lp_t
      - "weighted_multi": y[t] = sum_i w_i * (lp_{t+h_i} - lp_t)

    Các bước biến đổi:
      1. Tính y_raw (endpoint hoặc weighted_multi)
      2. Huberize (optional)
      3. Regime adjust theo volatility (optional)
      4. Regime adjust theo drawdown (optional)
      5. Drop các hàng không có y_direct (NaN ở cuối chuỗi)
    """

    df = df.copy()
    lp = df["lp"].values.astype(float)
    n = len(df)

    # ========== 1. Tính y_raw ==========
    y_raw = np.full(shape=n, fill_value=np.nan, dtype=float)

    if target_type == "endpoint":
        # Chuẩn 100d endpoint: lp_{t+H} - lp_t
        for t in range(n):
            if t + horizon < n:
                y_raw[t] = lp[t + horizon] - lp[t]

    elif target_type == "weighted_multi":
        horizons_arr = np.asarray(weighted_horizons, dtype=int)
        weights_arr = np.asarray(weighted_weights, dtype=float)

        if horizons_arr.shape[0] != weights_arr.shape[0]:
            raise ValueError("weighted_horizons và weighted_weights phải cùng độ dài")

        max_h = int(horizons_arr.max())

        # Chuẩn hóa weight để sum = 1
        s = weights_arr.sum()
        if not np.isclose(s, 1.0):
            weights_arr = weights_arr / s

        for t in range(n):
            if t + max_h < n:
                base_lp = lp[t]
                returns = lp[t + horizons_arr] - base_lp  # vector r20, r50, r100
                y_raw[t] = float(np.dot(weights_arr, returns))

    else:
        raise ValueError(f"Unknown target_type = {target_type}")

    # Nếu không có điểm nào đủ future thì trả ra df rỗng
    if np.all(np.isnan(y_raw)):
        df["y_direct"] = y_raw
        df_out = df.loc[[]].reset_index(drop=True)
        y_out = df_out["y_direct"].copy()
        return df_out, y_out

    y = y_raw.copy()

    # ========== 2. Huberize (clip extreme returns) ==========
    if huberize:
        valid = ~np.isnan(y)
        if valid.sum() > 0:
            low_p, high_p = huber_percentiles
            low, high = np.percentile(y[valid], [low_p, high_p])
            y[valid] = np.clip(y[valid], low, high)

    # ========== 3. Regime adjust theo volatility ==========
    if regime_vol_adjust:
        if regime_vol_col not in df.columns:
            raise KeyError(f"regime_vol_col '{regime_vol_col}' không có trong df")

        vol = df[regime_vol_col].values.astype(float)
        valid = ~np.isnan(y) & ~np.isnan(vol)

        if valid.sum() > 0:
            vol_thr = float(np.quantile(vol[valid], regime_vol_quantile))
            high_vol_mask = valid & (vol > vol_thr)
            # shrink khi vol spike
            y[high_vol_mask] = y[high_vol_mask] * regime_vol_shrink

    # ========== 4. Regime adjust theo drawdown (so với đỉnh lịch sử) ==========
    if regime_dd_adjust:
        if regime_dd_price_col in df.columns:
            price = df[regime_dd_price_col].values.astype(float)
        else:
            # fallback: dùng exp(lp) nếu không có cột close
            price = np.exp(lp)

        # max chạy từ đầu chuỗi tới hiện tại (ATH đến thời điểm t)
        roll_max = pd.Series(price).cummax().values
        dd = (roll_max - price) / roll_max  # drawdown fraction

        valid = ~np.isnan(y) & ~np.isnan(dd)
        if valid.sum() > 0:
            high_dd_mask = valid & (dd > regime_dd_threshold)
            y[high_dd_mask] = y[high_dd_mask] * regime_dd_shrink

    # ========== 5. Gán vào df và drop các hàng không có label ==========
    df["y_direct"] = y
    mask = ~np.isnan(df["y_direct"])
    df_out = df.loc[mask].reset_index(drop=True)
    y_out = df_out["y_direct"].copy()

    return df_out, y_out