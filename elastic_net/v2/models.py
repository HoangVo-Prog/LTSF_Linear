from typing import Dict, Tuple, Iterable, List

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error

from elastic_net.v2.config import CONFIG


# =========================================
# Hàm tạo ElasticNet
# =========================================

def _make_elasticnet(alpha: float, l1_ratio: float, random_state: int) -> ElasticNet:
    """
    Tạo một model ElasticNet với cấu hình chung.
    """
    return ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        random_state=random_state,
        max_iter=CONFIG["elastic_max_iter"],
    )


# =========================================
# Rolling training - mỗi điểm thời gian train trên quá khứ rồi dự đoán bước tiếp
# =========================================

def rolling_elasticnet_forecast(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    alpha: float,
    l1_ratio: float,
    window_type: str = "sliding",
    random_state: int = 42,
) -> np.ndarray:
    """
    Dự đoán 1-step-ahead theo kiểu rolling:
      - Với mỗi index i:
        + Chọn một cửa sổ dữ liệu quá khứ (sliding hoặc expanding)
        + Fit ElasticNet trên cửa sổ đó
        + Dự đoán y[i] từ X[i]
    Trả về mảng cùng chiều với y, các phần không đủ cửa sổ sẽ là nan.
    """
    n_samples = len(y)
    preds = np.full(n_samples, np.nan, dtype=float)

    if window_type not in {"sliding", "expanding"}:
        raise ValueError(f"Unknown window_type: {window_type}")

    for i in range(window_size, n_samples):
        if window_type == "sliding":
            train_slice = slice(i - window_size, i)
        else:
            train_slice = slice(0, i)

        X_train = X[train_slice]
        y_train = y[train_slice]

        # Nếu y_train không có đủ biến thiên thì không nên fit
        if len(np.unique(y_train)) < 2:
            continue

        model = _make_elasticnet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        model.fit(X_train, y_train)
        preds[i] = model.predict(X[i : i + 1])[0]

    return preds


def fit_final_elasticnet(
    X_scaled: np.ndarray,
    y: np.ndarray,
    window_size: int,
    alpha: float,
    l1_ratio: float,
    window_type: str = "sliding",
    random_state: int = 42,
) -> ElasticNet:
    """
    Fit model cuối cùng trên toàn bộ dữ liệu:
      - Nếu window_type là sliding thì chỉ lấy window_size điểm cuối
      - Nếu expanding thì dùng toàn bộ dữ liệu
    """
    n_samples = len(y)
    if n_samples < window_size:
        raise ValueError("Not enough samples to fit final ElasticNet")

    if window_type == "sliding":
        start = n_samples - window_size
        X_train = X_scaled[start:]
        y_train = y[start:]
    else:
        X_train = X_scaled
        y_train = y

    model = _make_elasticnet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    model.fit(X_train, y_train)
    return model


# =========================================
# Đánh giá dự đoán
# =========================================

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính MSE và MAE."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {"mse": mse, "mae": mae}


# =========================================
# Grid search cho ElasticNet
# =========================================

def _grid_search_elasticnet_mse(
    X_all_scaled: np.ndarray,
    y_all: np.ndarray,
    val_mask: np.ndarray,
    window_sizes: Iterable[int],
    window_types: Iterable[str],
    alphas: Iterable[float],
    l1_ratios: Iterable[float],
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Grid search trên:
      - window_size
      - window_type (sliding hoặc expanding)
      - alpha
      - l1_ratio

    Mỗi cấu hình:
      - Chạy rolling forecast trên toàn bộ chuỗi
      - Chỉ tính MSE MAE trên phần val_mask
    """
    records: List[Dict[str, float]] = []

    for w in window_sizes:
        for wt in window_types:
            for a in alphas:
                for l1 in l1_ratios:
                    preds_all = rolling_elasticnet_forecast(
                        X=X_all_scaled,
                        y=y_all,
                        window_size=w,
                        alpha=a,
                        l1_ratio=l1,
                        window_type=wt,
                        random_state=seed,
                    )
                    mask = val_mask & ~np.isnan(preds_all)
                    n_val = int(mask.sum())
                    if n_val == 0:
                        mse = np.nan
                        mae = np.nan
                    else:
                        metrics = evaluate_predictions(y_all[mask], preds_all[mask])
                        mse = metrics["mse"]
                        mae = metrics["mae"]

                    records.append(
                        {
                            "window_size": w,
                            "window_type": wt,
                            "alpha": a,
                            "l1_ratio": l1,
                            "n_val": n_val,
                            "mse": mse,
                            "mae": mae,
                        }
                    )

    df = pd.DataFrame(records)
    df = df.sort_values(["mse", "mae"], ascending=[True, True]).reset_index(drop=True)
    best_row = df.iloc[0].to_dict()
    best_config = {
        "window_size": int(best_row["window_size"]),
        "window_type": best_row["window_type"],
        "alpha": float(best_row["alpha"]),
        "l1_ratio": float(best_row["l1_ratio"]),
    }
    return df, best_config


def grid_search_window_and_reg(
    X_all_scaled: np.ndarray,
    y_all: np.ndarray,
    val_mask: np.ndarray,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Bước 1: grid search thô để chọn:
      - window_size
      - window_type
      - alpha
      - l1_ratio
    """
    window_sizes = [126, 252, 504, 756]
    window_types = ["sliding", "expanding"]
    alphas = [0.0005, 0.001, 0.005, 0.01]
    l1_ratios = [0.2, 0.5, 0.8]

    return _grid_search_elasticnet_mse(
        X_all_scaled=X_all_scaled,
        y_all=y_all,
        val_mask=val_mask,
        window_sizes=window_sizes,
        window_types=window_types,
        alphas=alphas,
        l1_ratios=l1_ratios,
        seed=seed,
    )


def grid_search_alpha_l1(
    X_all_scaled: np.ndarray,
    y_all: np.ndarray,
    val_mask: np.ndarray,
    base_window_size: int,
    base_window_type: str,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Bước 2: cố định window_size và window_type tốt nhất,
    chỉ tinh chỉnh alpha và l1_ratio mịn hơn.
    """
    alphas = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02]
    l1_ratios = [0.2, 0.5, 0.8]

    return _grid_search_elasticnet_mse(
        X_all_scaled=X_all_scaled,
        y_all=y_all,
        val_mask=val_mask,
        window_sizes=[base_window_size],
        window_types=[base_window_type],
        alphas=alphas,
        l1_ratios=l1_ratios,
        seed=seed,
    )
