import random
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================
# Hàm tiện ích chung
# =========================================

def set_seed(seed: int = 42) -> None:
    """Đặt seed cho random và numpy để kết quả tái lặp."""
    random.seed(seed)
    np.random.seed(seed)


def find_data_path(filename: str) -> Path:
    """
    Tìm file input trong các đường dẫn kiểu Kaggle.
    Nếu không thấy thì fallback về thư mục hiện tại.
    """
    candidates = [
        Path("/kaggle/input/aio-2025-linear-forecasting-challenge") / filename,
        Path("../input/aio-2025-linear-forecasting-challenge") / filename,
        Path("/kaggle/input/linear-forecaseting-fpt") / filename,
        Path("LTSF_LINEAR/data/aio-2025-linear-forecasting-challenge") / filename,
        Path("./") / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {filename} in known locations")


def clip_by_quantiles(
    s: pd.Series,
    mask: pd.Series,
    lower_q: float,
    upper_q: float,
) -> pd.Series:
    """
    Winsorize một series theo quantile, dùng chỉ phần nằm trong mask để ước lượng ngưỡng.
    """
    subset = s[mask & s.notna() & np.isfinite(s)]
    low = subset.quantile(lower_q)
    high = subset.quantile(upper_q)
    return s.clip(lower=low, upper=high)


# =========================================
# Tính return cơ bản và winsorization
# =========================================

def add_base_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm các cột:
      - close_shift1
      - volume_shift1
      - ret_1d: log return 1 ngày của giá
      - vol_chg: log thay đổi volume (log((v+1)/(v_prev+1)))
    """
    out = df.copy()
    out["close_shift1"] = out["close"].shift(1)
    out["volume_shift1"] = out["volume"].shift(1)

    # Giá luôn dương, dùng log ratio trực tiếp
    out["ret_1d"] = np.log(out["close"] / out["close_shift1"])

    # Volume có thể bằng 0, dùng log((v+1)/(v_prev+1)) để tránh chia 0
    out["vol_chg"] = np.log((out["volume"] + 1) / (out["volume_shift1"] + 1))

    return out


def winsorize_returns(
    df: pd.DataFrame,
    val_start: str,
    lower_q: float,
    upper_q: float,
) -> pd.DataFrame:
    """
    Winsorize ret_1d và vol_chg dựa trên phân phối của phần dữ liệu train
    (tức là những ngày trước val_start).
    """
    out = df.copy()
    val_start_ts = pd.Timestamp(val_start)
    train_mask = out["time"] < val_start_ts

    out["ret_1d_clipped"] = clip_by_quantiles(
        out["ret_1d"], mask=train_mask, lower_q=lower_q, upper_q=upper_q
    )
    out["vol_chg_clipped"] = clip_by_quantiles(
        out["vol_chg"], mask=train_mask, lower_q=lower_q, upper_q=upper_q
    )
    return out
