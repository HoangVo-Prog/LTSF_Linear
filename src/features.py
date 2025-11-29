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


def compute_stoch_kd(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    """
    Stochastic Oscillator %K, %D.
    """
    lowest_low = low.rolling(window=window, min_periods=window).min()
    highest_high = high.rolling(window=window, min_periods=window).max()

    k = 100.0 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
    d = k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d = d.rolling(window=smooth_d, min_periods=smooth_d).mean()

    out = pd.DataFrame({"stoch_k_14": k, "stoch_d_14": d}, index=close.index)
    return out


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    Average True Range 14.
    """
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr


def compute_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Commodity Channel Index.
    """
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window=window, min_periods=window).mean()
    mad = (tp - sma_tp).abs().rolling(window=window, min_periods=window).mean()
    cci = (tp - sma_tp) / (0.015 * (mad + 1e-8))
    return cci


# def build_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Input: df đã có lp, ret_1d, vol_log, ret_1d_clip, vol_log_clip.
#     Output: df_feat gồm time + feature + có thể base series (lp, close, volume).
#     """
#     df = df.copy()

#     # Giá và return cơ bản
#     # --------------------
#     # base returns ở nhiều horizon
#     df["ret_2d"] = df["lp"].diff(2)
#     df["ret_3d"] = df["lp"].diff(3)
#     df["ret_5d"] = df["lp"].diff(5)
#     df["ret_10d"] = df["lp"].diff(10)
#     df["ret_20d"] = df["lp"].diff(20)
#     df["ret_30d"] = df["lp"].diff(30)
#     df["ret_60d"] = df["lp"].diff(60)
#     df["ret_120d"] = df["lp"].diff(120)

#     # Volatility (std của ret_1d)
#     for win in [5, 10, 20, 30, 60, 120]:
#         df[f"vol_{win}"] = _rolling_std(df["ret_1d"], win)

#     # Moving averages giá (trên close)
#     for win in [5, 10, 20, 30, 60, 90, 120, 200]:
#         df[f"sma_{win}"] = _rolling_mean(df["close"], win)

#     # EMA các horizon (trên close)
#     for span in [5, 12, 20, 26, 50]:
#         df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()

#     # Price relative vs SMA
#     for win in [5, 10, 20, 30, 60, 90, 120, 200]:
#         sma_col = f"sma_{win}"
#         if sma_col in df.columns:
#             df[f"price_sma{win}"] = df["close"] / (df[sma_col] + 1e-8)

#     # Drawdown các horizon
#     for win in [20, 60, 120, 200]:
#         rolling_max = df["close"].rolling(window=win, min_periods=win).max()
#         df[f"dd_{win}"] = df["close"] / (rolling_max + 1e-8) - 1.0

#     # Range based features nếu có high/low
#     if "high" in df.columns and "low" in df.columns:
#         df["hl_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)
#         df["hl_range_abs"] = (df["high"] - df["low"]).abs()

#     # Volume based
#     # ------------
#     vol = df["volume"]
#     for win in [5, 10, 20, 60]:
#         df[f"vol_sma_{win}"] = _rolling_mean(vol, win)
#         df[f"vol_norm_{win}"] = vol / (df[f"vol_sma_{win}"] + 1e-8)

#     df["vol_z_20"] = (vol - _rolling_mean(vol, 20)) / (_rolling_std(vol, 20) + 1e-8)
#     df["vol_vol_20"] = _rolling_std(df["vol_log"], 20)

#     # Dollar volume
#     df["dollar_vol"] = df["close"] * df["volume"]

#     # Corr return - volume (20 phiên)
#     df["corr_ret_vol_20"] = df["ret_1d"].rolling(window=20, min_periods=20).corr(df["vol_log"])

#     # RSI với nhiều window
#     df["rsi_14"] = compute_rsi(df["close"], window=14)
#     df["rsi_7"] = compute_rsi(df["close"], window=7)
#     df["rsi_28"] = compute_rsi(df["close"], window=28)

#     # Bollinger 20 và 60
#     for win in [20, 60]:
#         sma = _rolling_mean(df["close"], win)
#         std = _rolling_std(df["close"], win)
#         df[f"bb_up_{win}"] = sma + 2.0 * std
#         df[f"bb_low_{win}"] = sma - 2.0 * std
#         df[f"bb_width_{win}"] = (df[f"bb_up_{win}"] - df[f"bb_low_{win}"]) / (sma + 1e-8)

#     # MACD chuẩn 12-26-9 và thêm MACD 5-20-9
#     df["macd"] = df["ema_12"] - df["ema_26"]
#     df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
#     df["macd_hist"] = df["macd"] - df["macd_signal"]

#     df["macd_fast"] = df["ema_5"] - df["ema_20"]
#     df["macd_fast_signal"] = df["macd_fast"].ewm(span=9, adjust=False).mean()
#     df["macd_fast_hist"] = df["macd_fast"] - df["macd_fast_signal"]

#     # Stochastic, ATR, CCI nếu có high/low
#     if "high" in df.columns and "low" in df.columns:
#         stoch_df = compute_stoch_kd(df["close"], df["high"], df["low"], window=14)
#         df = pd.concat([df, stoch_df], axis=1)

#         df["atr_14"] = compute_atr(df["high"], df["low"], df["close"], window=14)
#         df["cci_20"] = compute_cci(df["high"], df["low"], df["close"], window=20)

#     # Calendar features
#     df["dow"] = df["time"].dt.dayofweek  # 0..6
#     df["month"] = df["time"].dt.month    # 1..12

#     # Option thêm sin-cos cho tháng
#     df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
#     df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

#     # Lag features
#     # ------------
#     # Chọn các feature "xịn" để lag
#     lag_cols = [
#         "ret_1d_clip",
#         "vol_log_clip",
#         "ret_5d",
#         "ret_20d",
#         "ret_60d",
#         "vol_5",
#         "vol_20",
#         "vol_60",
#         "vol_120",
#         "price_sma5",
#         "price_sma10",
#         "price_sma20",
#         "price_sma60",
#         "price_sma120",
#         "rsi_7",
#         "rsi_14",
#         "rsi_28",
#         "macd",
#         "macd_signal",
#         "macd_hist",
#         "macd_fast",
#         "macd_fast_hist",
#         "bb_width_20",
#         "bb_width_60",
#         "dd_20",
#         "dd_60",
#         "dd_120",
#         "vol_norm_20",
#         "vol_z_20",
#         "stoch_k_14",
#         "stoch_d_14",
#         "cci_20",
#         "atr_14",
#         "dollar_vol",
#         "corr_ret_vol_20",
#     ]

#     lags = [1, 2, 3, 5, 10, 20]

#     # Để tránh DataFrame fragmented, build lag trong dict, rồi concat một lần
#     lag_data = {}
#     for col in lag_cols:
#         if col not in df.columns:
#             continue
#         for lag in lags:
#             lag_name = f"{col}_lag{lag}"
#             lag_data[lag_name] = df[col].shift(lag)

#     if lag_data:
#         lag_df = pd.DataFrame(lag_data, index=df.index)
#         df = pd.concat([df, lag_df], axis=1)

#     # ======================================================
#     # 100d aligned features và long horizon trend on lp
#     # ======================================================

#     # 1) 100 ngày quá khứ: return, drift, volatility
#     df["ret_100d_back"] = df["lp"] - df["lp"].shift(100)
#     df["drift_100d_back"] = df["ret_100d_back"] / 100.0
#     df["vol_100d_back"] = _rolling_std(df["ret_1d"], 100)

#     # 2) Long horizon EMA/SMA trên lp (log price)
#     for win in [20, 60, 120]:
#         df[f"ema_lp_{win}"] = df["lp"].ewm(span=win, adjust=False).mean()
#         df[f"sma_lp_{win}"] = _rolling_mean(df["lp"], win)
#         df[f"lp_minus_ema_{win}"] = df["lp"] - df[f"ema_lp_{win}"]
#         df[f"lp_minus_sma_{win}"] = df["lp"] - df[f"sma_lp_{win}"]

#     # 3) Volatility ratio và vol-of-vol trên return
#     #    (vol_20, vol_60, vol_120 đã có ở trên)
#     if all(col in df.columns for col in ["vol_20", "vol_60", "vol_120"]):
#         df["vol_ratio_20_60"] = df["vol_20"] / (df["vol_60"] + 1e-8)
#         df["vol_ratio_60_120"] = df["vol_60"] / (df["vol_120"] + 1e-8)

#     abs_ret = df["ret_1d"].abs()
#     df["ret_vol_of_vol_20"] = _rolling_std(abs_ret, 20)

#     # 4) Vị trí trong range 60 và 120 ngày (giúp model hiểu cổ đang ở đáy hay đỉnh)
#     for win in [60, 120]:
#         roll_max = df["close"].rolling(window=win, min_periods=win).max()
#         roll_min = df["close"].rolling(window=win, min_periods=win).min()
#         df[f"pos_in_range_{win}"] = (df["close"] - roll_min) / (roll_max - roll_min + 1e-8)

#     # 5) Volume regime: ratio 20/60 và spike flag
#     if "vol_sma_20" in df.columns and "vol_sma_60" in df.columns:
#         df["vol_ma_ratio_20_60"] = df["vol_sma_20"] / (df["vol_sma_60"] + 1e-8)

#     # spike volume mạnh so với 20 ngày
#     df["is_vol_spike_20"] = (df["vol_z_20"] > 2.0).astype(float)

#     return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df đã có lp, ret_1d, vol_log, ret_1d_clip, vol_log_clip.
    Output: df_feat gồm time + feature + có thể base series (lp, close, volume).
    """
    df = df.copy()

    # Giá và return cơ bản
    # base returns ở nhiều horizon
    df["ret_2d"] = df["lp"].diff(2)
    df["ret_3d"] = df["lp"].diff(3)
    df["ret_5d"] = df["lp"].diff(5)
    df["ret_10d"] = df["lp"].diff(10)
    df["ret_20d"] = df["lp"].diff(20)
    df["ret_30d"] = df["lp"].diff(30)
    df["ret_60d"] = df["lp"].diff(60)
    df["ret_120d"] = df["lp"].diff(120)

    # Volatility (std của ret_1d)
    for win in [5, 10, 20, 30, 60, 120]:
        df[f"vol_{win}"] = _rolling_std(df["ret_1d"], win)

    # Moving averages giá
    for win in [5, 10, 20, 30, 60, 90, 120, 200]:
        df[f"sma_{win}"] = _rolling_mean(df["close"], win)

    # EMA các horizon
    for span in [5, 12, 20, 26, 50]:
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()

    # Price relative vs SMA
    for win in [5, 10, 20, 30, 60, 90, 120, 200]:
        sma_col = f"sma_{win}"
        if sma_col in df.columns:
            df[f"price_sma{win}"] = df["close"] / (df[sma_col] + 1e-8)

    # Drawdown các horizon
    for win in [20, 60, 120, 200]:
        rolling_max = df["close"].rolling(window=win, min_periods=win).max()
        df[f"dd_{win}"] = df["close"] / (rolling_max + 1e-8) - 1.0

    # Range based features nếu có high low
    if "high" in df.columns and "low" in df.columns:
        df["hl_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)
        df["hl_range_abs"] = (df["high"] - df["low"]).abs()

    # Volume based
    vol = df["volume"]
    for win in [5, 10, 20, 60]:
        df[f"vol_sma_{win}"] = _rolling_mean(vol, win)
        df[f"vol_norm_{win}"] = vol / (df[f"vol_sma_{win}"] + 1e-8)

    df["vol_z_20"] = (vol - _rolling_mean(vol, 20)) / (_rolling_std(vol, 20) + 1e-8)
    df["vol_vol_20"] = _rolling_std(df["vol_log"], 20)

    # Dollar volume
    df["dollar_vol"] = df["close"] * df["volume"]

    # Corr return volume 20 phiên
    df["corr_ret_vol_20"] = df["ret_1d"].rolling(window=20, min_periods=20).corr(df["vol_log"])

    # RSI với nhiều window
    df["rsi_14"] = compute_rsi(df["close"], window=14)
    df["rsi_7"] = compute_rsi(df["close"], window=7)
    df["rsi_28"] = compute_rsi(df["close"], window=28)

    # Bollinger 20 và 60
    for win in [20, 60]:
        sma = _rolling_mean(df["close"], win)
        std = _rolling_std(df["close"], win)
        df[f"bb_up_{win}"] = sma + 2.0 * std
        df[f"bb_low_{win}"] = sma - 2.0 * std
        df[f"bb_width_{win}"] = (df[f"bb_up_{win}"] - df[f"bb_low_{win}"]) / (sma + 1e-8)

    # MACD chuẩn 12 26 9 và MACD nhanh 5 20 9
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["macd_fast"] = df["ema_5"] - df["ema_20"]
    df["macd_fast_signal"] = df["macd_fast"].ewm(span=9, adjust=False).mean()
    df["macd_fast_hist"] = df["macd_fast"] - df["macd_fast_signal"]

    # Stochastic, ATR, CCI nếu có high low
    if "high" in df.columns and "low" in df.columns:
        stoch_df = compute_stoch_kd(df["close"], df["high"], df["low"], window=14)
        df = pd.concat([df, stoch_df], axis=1)

        df["atr_14"] = compute_atr(df["high"], df["low"], df["close"], window=14)
        df["cci_20"] = compute_cci(df["high"], df["low"], df["close"], window=20)

    # Calendar features
    df["dow"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Lag features
    lag_cols = [
        "ret_1d_clip",
        "vol_log_clip",
        "ret_5d",
        "ret_20d",
        "ret_60d",
        "vol_5",
        "vol_20",
        "vol_60",
        "vol_120",
        "price_sma5",
        "price_sma10",
        "price_sma20",
        "price_sma60",
        "price_sma120",
        "rsi_7",
        "rsi_14",
        "rsi_28",
        "macd",
        "macd_signal",
        "macd_hist",
        "macd_fast",
        "macd_fast_hist",
        "bb_width_20",
        "bb_width_60",
        "dd_20",
        "dd_60",
        "dd_120",
        "vol_norm_20",
        "vol_z_20",
        "stoch_k_14",
        "stoch_d_14",
        "cci_20",
        "atr_14",
        "dollar_vol",
        "corr_ret_vol_20",
    ]

    lags = [1, 2, 3, 5, 10, 20]

    lag_data = {}
    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in lags:
            lag_name = f"{col}_lag{lag}"
            lag_data[lag_name] = df[col].shift(lag)

    if lag_data:
        lag_df = pd.DataFrame(lag_data, index=df.index)
        df = pd.concat([df, lag_df], axis=1)

    return df
