# features.py

import numpy as np
import pandas as pd
from typing import List, Tuple
from config import MAX_LAG, RET_WINDOWS, VOL_WINDOWS, SMA_WINDOWS


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # basic returns
    df["ret_1d"] = df["close"].pct_change()
    df["log_ret_1d"] = df["log_price"].diff()

    for w in RET_WINDOWS:
        df[f"log_ret_{w}d"] = df["log_price"].diff(w)
        df[f"ret_{w}d"] = df["close"].pct_change(w)

    # rolling volatility on log returns
    for w in VOL_WINDOWS:
        df[f"vol_{w}d"] = df["log_ret_1d"].rolling(w).std()

    # SMA and price vs SMA
    for w in SMA_WINDOWS:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"price_sma_{w}_rel"] = df["close"] / df[f"sma_{w}"] - 1.0

    # volume features
    for w in SMA_WINDOWS:
        df[f"vol_ma_{w}"] = df["volume"].rolling(w).mean()
        df[f"vol_rel_{w}"] = df["volume"] / (df[f"vol_ma_{w}"] + 1e-8) - 1.0

    # simple RSI like feature using price changes
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    window = 14
    roll_up = up.rolling(window).mean()
    roll_down = down.rolling(window).mean()
    rs = roll_up / (roll_down + 1e-8)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    return df


def make_supervised_residual_dataset(
    df: pd.DataFrame,
    max_lag: int = MAX_LAG,
    target_col: str = "resid",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Uses df with trend and resid already added.
    Target is next day residual: resid_{t+1}.
    Features are lagged versions of all chosen base columns.
    """

    df = df.copy()
    # define which base features to lag
    base_cols = [
        "resid",
        "log_price",
        "ret_1d",
        "log_ret_1d",
    ]

    # add technical feature names
    for w in RET_WINDOWS:
        base_cols += [f"log_ret_{w}d", f"ret_{w}d"]
    for w in VOL_WINDOWS:
        base_cols += [f"vol_{w}d"]
    for w in SMA_WINDOWS:
        base_cols += [f"sma_{w}", f"price_sma_{w}_rel", f"vol_ma_{w}", f"vol_rel_{w}"]
    base_cols += ["rsi_14"]

    base_cols = [c for c in base_cols if c in df.columns]

    # target
    df["target_resid_next"] = df[target_col].shift(-1)

    feature_frames = []
    feature_names: List[str] = []

    for col in base_cols:
        for lag in range(max_lag + 1):
            cname = f"{col}_lag{lag}"
            feature_names.append(cname)
            feature_frames.append(df[col].shift(lag).rename(cname))

    X = pd.concat(feature_frames, axis=1)
    y = df["target_resid_next"]

    # drop rows with NaN
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    return X, y, feature_names
