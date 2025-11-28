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
    avg_gain = gain.mean()
    avg_loss = loss.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def compute_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Tính RSI rolling cho toàn bộ series giá.
    """
    arr = close.values.astype(float)
    rsi = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period + 1:
        return pd.Series(rsi, index=close.index)
    for i in range(period, len(arr)):
        window = arr[i - period : i + 1]
        rsi[i] = _rsi_from_window(window, period)
    return pd.Series(rsi, index=close.index)


# =========================================
# Build toàn bộ feature để train model
# =========================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df đã có các cột:
      - time
      - close
      - ret_1d_clipped

    Output: dataframe có
      - time
      - toàn bộ FEATURE_NAMES
      - y: target là return ngày t+1
    """
    # Lấy các cột cơ bản
    feat = df[["time", "close", "ret_1d_clipped"]].copy()

    r = feat["ret_1d_clipped"]
    c = feat["close"]

    # Lags của return 1 đến 10 ngày
    for lag in range(1, 11):
        if lag == 1:
            feat[f"ret_lag{lag}"] = r
        else:
            feat[f"ret_lag{lag}"] = r.shift(lag - 1)

    # Độ biến động rolling của return (ngắn đến dài)
    feat["vol_5"] = r.rolling(5).std()
    feat["vol_10"] = r.rolling(10).std()
    feat["vol_20"] = r.rolling(20).std()
    feat["vol_60"] = r.rolling(60).std()
    feat["vol_100"] = r.rolling(100).std()

    # Min, max return trong 20 ngày gần nhất
    feat["ret_roll_min_20"] = r.rolling(20).min()
    feat["ret_roll_max_20"] = r.rolling(20).max()

    # Z score của return so với rolling mean, std 20 ngày
    roll_mean_20 = r.rolling(20).mean()
    roll_std_20 = r.rolling(20).std()
    feat["ret_z_20"] = (r - roll_mean_20) / roll_std_20.replace(0, np.nan)

    # Trung bình return ngắn và trung hạn
    feat["mean_ret_5"] = r.rolling(5).mean()
    feat["mean_ret_10"] = r.rolling(10).mean()
    feat["mean_ret_20"] = roll_mean_20
    feat["mean_ret_60"] = r.rolling(60).mean()
    feat["mean_ret_100"] = r.rolling(100).mean()

    # SMA và trend theo SMA ở nhiều horizon
    feat["sma10"] = c.rolling(10).mean()
    feat["sma20"] = c.rolling(20).mean()
    feat["sma50"] = c.rolling(50).mean()
    feat["sma100"] = c.rolling(100).mean()

    feat["price_trend_10"] = (c - feat["sma10"]) / feat["sma10"]
    feat["price_trend_20"] = (c - feat["sma20"]) / feat["sma20"]
    feat["price_trend_50"] = (c - feat["sma50"]) / feat["sma50"]
    feat["price_trend_100"] = (c - feat["sma100"]) / feat["sma100"]

    # Drawdown và days since max trên 100 ngày
    roll_max_100 = c.rolling(100).max()
    feat["drawdown_100"] = (c - roll_max_100) / roll_max_100

    # days_since_max_100: số ngày kể từ đỉnh gần nhất trong cửa sổ 100 ngày
    feat["days_since_max_100"] = c.rolling(100).apply(
        lambda w: len(w) - 1 - int(np.argmax(w)), raw=True
    )

    # RSI 14 ngày
    feat["rsi_14"] = compute_rsi_series(c, period=14)

    # Bollinger band width 20 ngày
    std20_price = c.rolling(20).std()
    upper20 = feat["sma20"] + 2 * std20_price
    lower20 = feat["sma20"] - 2 * std20_price
    feat["bb_width_20"] = (upper20 - lower20) / feat["sma20"]

    # Feature theo lịch
    feat["dow"] = feat["time"].dt.dayofweek.astype(int)
    feat["month"] = feat["time"].dt.month.astype(int)

    # Target: return ngày t+1
    feat["y"] = feat["ret_1d_clipped"].shift(-1)

    cols = ["time", "close"] + FEATURE_NAMES + ["y"]
    feat = feat[cols]

    # Bỏ các hàng không đủ dữ liệu rolling
    feat = feat.dropna().reset_index(drop=True)
    return feat
