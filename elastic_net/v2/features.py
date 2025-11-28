import numpy as np
import pandas as pd

from config import FEATURE_NAMES


# =========================================
# Tính RSI từ chuỗi giá
# =========================================

def _rsi_from_window(prices_window: np.ndarray, period: int) -> float:
    """
    Tính RSI từ một cửa sổ giá theo công thức trung bình gain và loss.
    """
    delta = np.diff(prices_window)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    if len(gain) < period:
        return 50.0

    avg_gain = gain[-period:].mean()
    avg_loss = loss[-period:].mean()

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return float(rsi)


def compute_rsi_series(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Tính RSI cho từng ngày bằng cách trượt cửa sổ.
    """
    prices_arr = prices.values.astype(float)
    out = np.full(len(prices_arr), np.nan, dtype=float)

    for i in range(period, len(prices_arr)):
        window = prices_arr[i - period : i + 1]
        out[i] = _rsi_from_window(window, period)

    return pd.Series(out, index=prices.index)


# =========================================
# Build technical features từ df đã có ret_1d, ret_1d_clipped
# =========================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Từ df cột [time, close, volume, ret_1d, vol_chg, ret_1d_clipped, vol_chg_clipped]
    tạo ra bảng feature với các cột FEATURE_NAMES và target y.

    Thiết kế mới:
      - Input feature dùng ret_1d_clipped (winsorized) để giảm outlier.
      - Target y dùng ret_1d (unclipped) dịch 1 ngày.
      - Bỏ hoàn toàn các feature dựa trên volume.
      - Thêm các feature chậm 60, 120, 252 ngày cho regime dài hạn.
    """
    feat = pd.DataFrame(index=df.index.copy())
    feat["time"] = df["time"].copy()

    # Chuỗi giá và return winsorized
    c = df["close"].astype(float)
    r = df["ret_1d_clipped"].astype(float)

    # 1. Base return features
    feat["ret_1d_clipped"] = r

    for k in range(1, 11):
        feat[f"ret_lag{k}"] = r.shift(k)

    # 2. Rolling volatility và thống kê ngắn hạn
    feat["vol_5"] = r.rolling(5).std()
    feat["vol_10"] = r.rolling(10).std()
    feat["vol_20"] = r.rolling(20).std()

    feat["ret_roll_min_20"] = r.rolling(20).min()
    feat["ret_roll_max_20"] = r.rolling(20).max()

    roll_mean_20 = r.rolling(20).mean()
    roll_std_20 = r.rolling(20).std()
    feat["ret_z_20"] = (r - roll_mean_20) / roll_std_20.replace(0, np.nan)

    feat["mean_ret_5"] = r.rolling(5).mean()
    feat["mean_ret_10"] = r.rolling(10).mean()
    feat["mean_ret_20"] = roll_mean_20

    # 3. SMA và trend theo SMA
    feat["sma10"] = c.rolling(10).mean()
    feat["sma20"] = c.rolling(20).mean()
    feat["price_trend_10"] = (c - feat["sma10"]) / feat["sma10"]
    feat["price_trend_20"] = (c - feat["sma20"]) / feat["sma20"]

    # 4. RSI 14 ngày
    feat["rsi_14"] = compute_rsi_series(c, period=14)

    # 5. Bollinger band width 20 ngày
    std20_price = c.rolling(20).std()
    upper = feat["sma20"] + 2.0 * std20_price
    lower = feat["sma20"] - 2.0 * std20_price
    feat["bb_width_20"] = (upper - lower) / feat["sma20"]

    # 6. Feature chậm cho regime 60, 120, 252 ngày
    # Dùng ret_1d_clipped để nhất quán với input training
    # Cumulative log return
    feat["cumret_60"] = r.rolling(60).sum()
    feat["cumret_120"] = r.rolling(120).sum()
    feat["cumret_252"] = r.rolling(252).sum()

    # Realized volatility
    feat["realized_vol_60"] = r.rolling(60).std()
    feat["realized_vol_120"] = r.rolling(120).std()

    # Drawdown so với đỉnh gần nhất trong 60, 120 ngày
    rolling_max_60 = c.rolling(60).max()
    rolling_max_120 = c.rolling(120).max()
    feat["drawdown_60"] = (c / rolling_max_60) - 1.0
    feat["drawdown_120"] = (c / rolling_max_120) - 1.0

    # Vị trí phần trăm giá trong khoảng 252 ngày (giống 52 week high/low)
    rolling_min_252 = c.rolling(252).min()
    rolling_max_252 = c.rolling(252).max()
    feat["price_pct_252"] = (c - rolling_min_252) / (rolling_max_252 - rolling_min_252)

    # 7. Feature theo lịch
    feat["dow"] = feat["time"].dt.dayofweek.astype(int)
    feat["month"] = feat["time"].dt.month.astype(int)

    # Target: return ngày t+1, dùng ret_1d (unclipped)
    feat["y"] = df["ret_1d"].shift(-1)

    # Chọn đúng thứ tự cột
    cols = ["time"] + FEATURE_NAMES + ["y"]
    feat = feat[cols]

    # Bỏ các hàng không đủ dữ liệu rolling
    feat = feat.dropna().reset_index(drop=True)
    return feat
