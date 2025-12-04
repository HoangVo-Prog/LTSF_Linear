# patchtst/postprocessing.py

from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from config import HORIZON
from patchtst_model import build_patchtst_model, make_neuralforecast
from metrics import compute_metrics


def collect_postprocessing_data(
    train_nf_full,
    best_params,
    n_splits: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Chạy cross validation thời gian để thu X_post, y_post."""

    tscv = TimeSeriesSplit(n_splits=n_splits)
    X_post = []
    y_post = []

    full_data = train_nf_full.copy()

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(full_data)):
        print(f" Fold {fold_idx} của TimeSeriesSplit")
        train_fold = full_data.iloc[train_idx]
        val_fold = full_data.iloc[val_idx]

        model_fold = build_patchtst_model(
            best_params,
            horizon=min(HORIZON, len(val_fold)),
        )
        nf_fold = make_neuralforecast(model_fold)
        nf_fold.fit(df=train_fold, val_size=0)
        forecast_fold = nf_fold.predict()
        pred_col = [c for c in forecast_fold.columns if c not in ["unique_id", "ds"]][0]

        pred_fold = forecast_fold[pred_col].values[: len(val_fold)]
        true_fold = val_fold["y"].values[: len(pred_fold)]

        X_post.extend(pred_fold.reshape(-1, 1))
        y_post.extend(true_fold)

    X_post = np.array(X_post)
    y_post = np.array(y_post)
    print(f" Đã thu thập {len(X_post)} điểm cho post processing")
    return X_post, y_post


def train_linear_post_model(
    X_post: np.ndarray,
    y_post: np.ndarray,
) -> LinearRegression:
    """Train Linear Regression để map pred baseline -> actual."""
    model = LinearRegression()
    model.fit(X_post, y_post)
    return model


def apply_postprocessing(
    baseline_pred: np.ndarray,
    post_model: LinearRegression,
    y_true: np.ndarray,
):
    """Áp dụng post processing và tính metrics."""
    pred_post = post_model.predict(baseline_pred.reshape(-1, 1))
    metrics = compute_metrics(y_true, pred_post)
    coef = float(post_model.coef_[0])
    intercept = float(post_model.intercept_)

    print("\n Post processing (Linear Regression):")
    print(f" - Formula: y = {coef:.4f} * pred + {intercept:.4f}")

    return pred_post, metrics, coef, intercept
