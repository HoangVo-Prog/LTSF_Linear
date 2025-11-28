from typing import List
import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """
    Đọc csv, parse time, sort theo time.
    Yêu cầu cột: time, close, volume.
    """
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def add_base_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm các series cơ bản: lp, ret_1d, vol_raw, vol_log.
    """
    df = df.copy()
    df["lp"] = np.log(df["close"])
    df["ret_1d"] = df["lp"].diff(1)
    df["vol_raw"] = df["volume"]
    df["vol_log"] = np.log((df["volume"] + 1.0) / (df["volume"].shift(1) + 1.0))
    return df


def winsorize_series(
    df: pd.DataFrame,
    train_mask: pd.Series,
    cols: List[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    suffix: str = "_clip",
) -> pd.DataFrame:
    """
    Clip các cột theo quantile trên train period.
    Tạo thêm cột <col><suffix>.
    """
    df = df.copy()
    train_df = df.loc[train_mask]

    for col in cols:
        lo = train_df[col].quantile(lower_q)
        hi = train_df[col].quantile(upper_q)
        clipped = df[col].clip(lower=lo, upper=hi)
        df[f"{col}{suffix}"] = clipped

    return df


def ensure_business_day_indexing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đảm bảo time sorted, không duplicated.
    Không ép về full business calendar, chỉ check basic.
    """
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)
    if df["time"].duplicated().any():
        raise ValueError("Duplicated time values detected")
    return df
