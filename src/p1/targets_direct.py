from typing import Tuple
import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HORIZON


def _apply_robust_asymmetric_huber(
    y: np.ndarray,
    outer_percentiles=(0.5, 99.5),
    k_pos: float = 0.8,
    k_neg: float = 2.0,
) -> np.ndarray:
    """
    Huber cải tiến, bất đối xứng:

    - Clip nhẹ theo outer_percentiles để bỏ extreme điên.
    - Dùng median + MAD để chuẩn hoá.
    - Clip z-score: phía dương bị chặn sớm (k_pos nhỏ),
      phía âm được phép lớn hơn (k_neg lớn)
      -> giảm bias tăng nhưng vẫn giữ được cú rơi mạnh.
    """
    y_new = y.copy()
    valid = ~np.isnan(y_new)
    if valid.sum() == 0:
        return y_new

    v = y_new[valid]

    # 1) Outer clip nhẹ
    low_o, high_o = np.percentile(v, outer_percentiles)
    v = np.clip(v, low_o, high_o)

    # 2) Robust center + scale
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad <= 1e-8:
        std = np.std(v)
        scale = std if std > 1e-8 else 1.0
    else:
        scale = mad * 1.4826  # MAD ~ std

    z = (y_new - med) / scale

    # 3) Clip bất đối xứng
    z_clipped = z.copy()
    pos_mask = z > 0
    neg_mask = z < 0
    z_clipped[pos_mask] = np.minimum(z[pos_mask], k_pos)
    z_clipped[neg_mask] = np.maximum(z[neg_mask], -k_neg)

    y_new = med + z_clipped * scale
    return y_new


def build_direct_100d_target(
    df: pd.DataFrame,
    horizon: int = HORIZON,

    # loại target
    target_type: str = "endpoint",  # "endpoint", "weighted_multi", "ms_momentum", "dir_momentum"

    # endpoint 100d: không cần thêm param

    # weighted_multi: thường cho 20, 50, 100
    weighted_horizons=(20, 50, 100),
    weighted_weights=(0.5, 0.3, 0.2),

    # multi scale momentum: mặc định 0.7*r10 + 0.2*r20 + 0.1*r50
    ms_horizons=(10, 20, 50),
    ms_weights=(0.7, 0.2, 0.1),

    # directional momentum: dùng r10, r20, r50
    dir_h_short: int = 10,
    dir_h_mid: int = 20,
    dir_h_long: int = 50,
    dir_mag_div_mid: float = 2.0,
    dir_mag_div_long: float = 3.0,

    # Huber cải tiến
    huberize: bool = True,
    huber_outer_percentiles=(0.5, 99.5),
    huber_k_pos: float = 0.8,
    huber_k_neg: float = 2.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Tạo target y_direct cho nhiều mode khác nhau:

    target_type:
      - "endpoint":
          y[t] = lp_{t+H} - lp_t
      - "weighted_multi":
          y[t] = sum_i w_i * (lp_{t+h_i} - lp_t)
          dùng weighted_horizons, weighted_weights
      - "ms_momentum":
          y[t] = sum_i w_i * (lp_{t+h_i} - lp_t)
          mặc định 0.7*r10 + 0.2*r20 + 0.1*r50
      - "dir_momentum":
          r10, r20, r50
          mag_candidates = [|r10|, |r20| / dir_mag_div_mid, |r50| / dir_mag_div_long]
          mag = min(mag_candidates)
          sign ưu tiên r10, nếu r10 = 0 thì dùng r20, nếu vẫn 0 thì 0
          y[t] = sign * mag

    Sau đó, nếu huberize=True thì áp dụng Huber cải tiến lên toàn bộ y.
    """

    df = df.copy()
    lp = df["lp"].values.astype(float)
    n = len(df)

    # ===== 1. Tính y_raw theo target_type =====
    y_raw = np.full(shape=n, fill_value=np.nan, dtype=float)

    if target_type == "endpoint":
        for t in range(n):
            if t + horizon < n:
                y_raw[t] = lp[t + horizon] - lp[t]

    elif target_type == "weighted_multi":
        horizons_arr = np.asarray(weighted_horizons, dtype=int)
        weights_arr = np.asarray(weighted_weights, dtype=float)

        if horizons_arr.shape[0] != weights_arr.shape[0]:
            raise ValueError("weighted_horizons và weighted_weights phải cùng độ dài")

        max_h = int(horizons_arr.max())

        s = weights_arr.sum()
        if not np.isclose(s, 1.0):
            weights_arr = weights_arr / s

        for t in range(n):
            if t + max_h < n:
                base_lp = lp[t]
                returns = lp[t + horizons_arr] - base_lp
                y_raw[t] = float(np.dot(weights_arr, returns))

    elif target_type == "ms_momentum":
        horizons_arr = np.asarray(ms_horizons, dtype=int)
        weights_arr = np.asarray(ms_weights, dtype=float)

        if horizons_arr.shape[0] != weights_arr.shape[0]:
            raise ValueError("ms_horizons và ms_weights phải cùng độ dài")

        max_h = int(horizons_arr.max())

        s = weights_arr.sum()
        if not np.isclose(s, 1.0):
            weights_arr = weights_arr / s

        for t in range(n):
            if t + max_h < n:
                base_lp = lp[t]
                returns = lp[t + horizons_arr] - base_lp
                y_raw[t] = float(np.dot(weights_arr, returns))

    elif target_type == "dir_momentum":
        max_h = max(dir_h_short, dir_h_mid, dir_h_long)

        for t in range(n):
            if t + max_h < n:
                base_lp = lp[t]
                r_short = lp[t + dir_h_short] - base_lp
                r_mid = lp[t + dir_h_mid] - base_lp
                r_long = lp[t + dir_h_long] - base_lp

                mag_candidates = [
                    abs(r_short),
                    abs(r_mid) / dir_mag_div_mid,
                    abs(r_long) / dir_mag_div_long,
                ]
                mag = float(np.min(mag_candidates))

                # direction ưu tiên r_short
                if r_short > 0:
                    sign = 1.0
                elif r_short < 0:
                    sign = -1.0
                elif r_mid > 0:
                    sign = 1.0
                elif r_mid < 0:
                    sign = -1.0
                else:
                    sign = 0.0

                y_raw[t] = sign * mag

    else:
        raise ValueError(f"Unknown target_type = {target_type}")

    # Không có future nào
    if np.all(np.isnan(y_raw)):
        df["y_direct"] = y_raw
        df_out = df.loc[[]].reset_index(drop=True)
        y_out = df_out["y_direct"].copy()
        return df_out, y_out

    y = y_raw.copy()

    # ===== 2. Huber cải tiến =====
    if huberize:
        y = _apply_robust_asymmetric_huber(
            y,
            outer_percentiles=huber_outer_percentiles,
            k_pos=huber_k_pos,
            k_neg=huber_k_neg,
        )

    # ===== 3. Gán vào df và drop các hàng không đủ future =====
    df["y_direct"] = y
    mask = ~np.isnan(df["y_direct"])
    df_out = df.loc[mask].reset_index(drop=True)
    y_out = df_out["y_direct"].copy()

    return df_out, y_out
