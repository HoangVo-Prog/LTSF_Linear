from typing import List
import numpy as np
import pandas as pd


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).std()


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI đơn giản trên giá đóng cửa.
    """
    delta = series.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df đã có lp, ret_1d, vol_log, ret_1d_clip, vol_log_clip.
    Output: df_feat gồm time + feature + có thể base series (lp, close).
    """
    df = df.copy()

    # Base price series
    df["ret_5d"] = df["lp"].diff(5)
    df["ret_20d"] = df["lp"].diff(20)

    # Volatility
    df["vol_5"] = _rolling_std(df["ret_1d"], 5)
    df["vol_20"] = _rolling_std(df["ret_1d"], 20)
    df["vol_60"] = _rolling_std(df["ret_1d"], 60)

    # Moving averages
    for win in [10, 20, 60, 120]:
        df[f"sma_{win}"] = _rolling_mean(df["close"], win)

    # EMAs cho MACD
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # Price relative
    df["price_sma10"] = df["close"] / df["sma_10"]
    df["price_sma60"] = df["close"] / df["sma_60"]
    df["price_sma120"] = df["close"] / df["sma_120"]

    # Drawdown
    rolling_max_60 = df["close"].rolling(window=60, min_periods=60).max()
    rolling_max_120 = df["close"].rolling(window=120, min_periods=120).max()
    df["dd_60"] = df["close"] / rolling_max_60 - 1.0
    df["dd_120"] = df["close"] / rolling_max_120 - 1.0

    # Volume based
    vol_mean_20 = _rolling_mean(df["volume"], 20)
    vol_std_20 = _rolling_std(df["volume"], 20)
    df["vol_norm_20"] = df["volume"] / vol_mean_20
    df["vol_z_20"] = (df["volume"] - vol_mean_20) / (vol_std_20 + 1e-8)
    df["vol_vol_20"] = _rolling_std(df["vol_log"], 20)

    # RSI
    df["rsi_14"] = compute_rsi(df["close"], window=14)

    # Bollinger bands
    sma_20 = df["sma_20"]
    std_20 = _rolling_std(df["close"], 20)
    df["bb_up_20"] = sma_20 + 2.0 * std_20
    df["bb_low_20"] = sma_20 - 2.0 * std_20
    df["bb_width_20"] = (df["bb_up_20"] - df["bb_low_20"]) / (sma_20 + 1e-8)

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Calendar features
    df["dow"] = df["time"].dt.dayofweek  # 0..6
    df["month"] = df["time"].dt.month    # 1..12

    # Lag features
    lag_cols = [
        "ret_1d_clip",
        "vol_log_clip",
        "vol_5",
        "vol_20",
        "vol_60",
        "price_sma10",
        "price_sma60",
        "price_sma120",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "dd_60",
        "dd_120",
    ]
    lags = [1, 2, 3, 5, 10, 20]

    for col in lag_cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Optional one hot calendar
    # df = pd.get_dummies(df, columns=["dow", "month"], drop_first=True)

    return df
