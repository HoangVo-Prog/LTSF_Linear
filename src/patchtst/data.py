# patchtst/data.py

import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config import (
    FPT_TRAIN_DRIVE_ID,
    FPT_TEST_DRIVE_ID,
    TRAIN_CSV_PATH,
    TEST_CSV_PATH,
    TARGET_COL,
    HORIZON,
)

def _ensure_gdown_installed() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError:
        subprocess.run(["pip", "install", "-q", "gdown"], check=True)


def download_if_missing(file_id: str, dst_path: Path) -> None:
    """Download file từ Google Drive nếu chưa tồn tại."""
    if dst_path.exists():
        print(f" File đã tồn tại: {dst_path}")
        return

    print(f" Đang download file từ Google Drive tới: {dst_path}")
    _ensure_gdown_installed()
    import gdown

    gdown.download(
        f"https://drive.google.com/uc?id={file_id}",
        str(dst_path),
        quiet=False,
    )

    if not dst_path.exists():
        raise RuntimeError(f" Download thất bại cho file {dst_path}")


def load_train_csv() -> pd.DataFrame:
    download_if_missing(FPT_TRAIN_DRIVE_ID, TRAIN_CSV_PATH)
    df = pd.read_csv(TRAIN_CSV_PATH, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(
        f" Đã load training data: {len(df)} điểm, "
        f"từ {df['time'].min()} đến {df['time'].max()}"
    )
    return df


def load_test_csv() -> pd.DataFrame:
    download_if_missing(FPT_TEST_DRIVE_ID, TEST_CSV_PATH)

    # đọc thô để xem có cột time không
    df_test_raw = pd.read_csv(
        TEST_CSV_PATH,
        parse_dates=["time"] if "time" in pd.read_csv(TEST_CSV_PATH, nrows=1).columns else None,
    )
    print(f" File test ban đầu: {len(df_test_raw):,} dòng")

    if "symbol" in df_test_raw.columns:
        df_test = df_test_raw[df_test_raw["symbol"] == "FPT"].copy()
        print(f" Sau khi lọc symbol FPT: {len(df_test):,} dòng")
    else:
        df_test = df_test_raw.copy()
        print(" Không có cột symbol, dùng toàn bộ file test")

    if "time" in df_test.columns:
        df_test = df_test.sort_values("time").reset_index(drop=True)

    return df_test


def split_train_val(
    df_train: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chia train  val  test_ground_truth (từ file test riêng)."""
    values = df_train[target_col].values.astype("float32")
    T = len(values)

    train_size = int(T * 0.8)
    val_size = int(T * 0.1)

    train_data = values[:train_size]
    val_data = values[train_size : train_size + val_size]
    rest_data = values[train_size + val_size :]

    print(" Chia dữ liệu training:")
    print(f" - Train: {len(train_data)} điểm")
    print(f" - Val:   {len(val_data)} điểm")
    print(f" - Còn lại: {len(rest_data)} điểm (không dùng trực tiếp)")

    return train_data, val_data, rest_data


def prepare_neuralforecast_frames(
    df_train: pd.DataFrame,
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chuẩn hóa thành 3 dataframe: train_nf, val_nf, train_nf_full."""
    from datetime import timedelta

    start_time = df_train["time"].iloc[0]
    train_nf = pd.DataFrame(
        {
            "unique_id": "FPT",
            "ds": pd.date_range(start=start_time, periods=len(train_data), freq="D"),
            "y": train_data,
        }
    )

    val_start_time = df_train["time"].iloc[len(train_data)]
    val_nf = pd.DataFrame(
        {
            "unique_id": "FPT",
            "ds": pd.date_range(start=val_start_time, periods=len(val_data), freq="D"),
            "y": val_data,
        }
    )

    train_nf_full = pd.concat([train_nf, val_nf], ignore_index=True)
    return train_nf, val_nf, train_nf_full


def get_test_ground_truth(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    horizon: int = HORIZON,
) -> np.ndarray:
    """Lấy y_true từ file test, lọc theo ngày sau training."""
    if "time" in df_test.columns:
        last_train_date = df_train["time"].max()
        print(f" Ngày cuối cùng trong train: {last_train_date.date()}")
        df_test = df_test[df_test["time"] > last_train_date].copy()
        df_test = df_test.sort_values("time").reset_index(drop=True)
        print(
            f" Sau khi lọc ngày > {last_train_date.date()}: {len(df_test)} dòng"
        )

    if len(df_test) >= horizon:
        y_true = df_test.head(horizon)["close"].values.astype("float32")
        print(f" Đã lấy {len(y_true)} điểm ground truth từ test data")
    elif len(df_test) > 0:
        y_true = df_test["close"].values.astype("float32")
        print(
            f" Đã lấy {len(y_true)} điểm ground truth (ít hơn horizon {horizon})"
        )
    else:
        raise RuntimeError(" Không có test data hợp lệ để lấy ground truth")

    return y_true
