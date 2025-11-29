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

def _apply_robust_asymmetric_huber(
    y: np.ndarray,
    outer_percentiles=(0.5, 99.5),
    k_pos: float = 0.8,
    k_neg: float = 2.0,
) -> np.ndarray:
    """
    Huber cải tiến, bất đối xứng:

    - Trước tiên clip rất nhẹ theo outer_percentiles để bỏ extreme insane.
    - Dùng median + MAD để chuẩn hoá.
    - Clip z-score: phía dương mạnh tay (k_pos nhỏ), phía âm nới (k_neg lớn)
      -> giảm bias tăng nhưng vẫn giữ được cú rơi mạnh.
    """
    y_new = y.copy()
    valid = ~np.isnan(y_new)
    if valid.sum() == 0:
        return y_new

    v = y_new[valid]

    # 1) Outer clip nhẹ để tránh vài điểm điên làm hỏng MAD
    low_o, high_o = np.percentile(v, outer_percentiles)
    v = np.clip(v, low_o, high_o)

    # 2) Robust center + scale
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad <= 1e-8:
        # fallback: dùng std nếu MAD quá nhỏ
        std = np.std(v)
        scale = std if std > 1e-8 else 1.0
    else:
        scale = mad * 1.4826  # MAD -> ~std

    z = (y_new - med) / scale

    # 3) Clip bất đối xứng: positive bị chặn sớm, negative được phép lớn hơn
    z_clipped = z.copy()
    z_clipped[z > 0] = np.minimum(z[z > 0], k_pos)
    z_clipped[z < 0] = np.maximum(z[z < 0], -k_neg)

    y_new = med + z_clipped * scale
    return y_new


def build_direct_100d_target(
    df: pd.DataFrame,
    horizon: int = HORIZON,
    target_type: str = "endpoint",      # "endpoint" hoặc "weighted_multi"
    weighted_horizons=(20, 50, 100),
    weighted_weights=(0.5, 0.3, 0.2),

    # Huber cải tiến
    huberize: bool = True,
    huber_outer_percentiles=(0.5, 99.5),
    huber_k_pos: float = 0.8,
    huber_k_neg: float = 2.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Tạo target y_direct:

    - endpoint: y[t] = lp_{t+H} - lp_t
    - weighted_multi: y[t] = sum_i w_i * (lp_{t+h_i} - lp_t)

    Sau đó áp dụng Huber cải tiến (robust + bất đối xứng) trên toàn bộ y_direct.
    """

    df = df.copy()
    lp = df["lp"].values.astype(float)
    n = len(df)

    # ===== 1. Tính y_raw =====
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

    else:
        raise ValueError(f"Unknown target_type = {target_type}")

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

    # ===== 3. Gán vào df & drop các hàng không có future =====
    df["y_direct"] = y
    mask = ~np.isnan(df["y_direct"])
    df_out = df.loc[mask].reset_index(drop=True)
    y_out = df_out["y_direct"].copy()

    return df_out, y_out

