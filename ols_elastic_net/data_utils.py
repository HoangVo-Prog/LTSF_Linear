# data_utils.py

import pandas as pd
import numpy as np 
from typing import Tuple
from config import TRAIN_CSV, TRAIN_END_DATE


def load_price_data(csv_path: str = TRAIN_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["t"] = df.index.astype(int)
    df["log_price"] = np.log(df["close"].astype(float) + 1e-8)
    return df


def train_val_split(df: pd.DataFrame, train_end: str = TRAIN_END_DATE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_date = pd.Timestamp(train_end)
    train_df = df[df["time"] < split_date].copy()
    val_df = df[df["time"] >= split_date].copy()
    return train_df, val_df
