# patchtst/metrics.py

from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    bias = float(np.mean(y_pred - y_true))

    return dict(
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        mape=mape,
        bias=bias,
    )


def print_metrics(label: str, metrics: Dict[str, float], baseline_mse: float = None):
    print(f"\n {label} Results:")
    print(f" - MSE:   {metrics['mse']:.4f}")
    print(f" - RMSE:  {metrics['rmse']:.4f}")
    print(f" - MAE:   {metrics['mae']:.4f}")
    print(f" - R2:    {metrics['r2']:.4f}")
    print(f" - MAPE:  {metrics['mape']:.2f} percent")
    print(f" - Bias:  {metrics['bias']:.4f}")

    if baseline_mse is not None:
        improvement = (baseline_mse - metrics["mse"]) / baseline_mse * 100.0
        print(f" - Cải thiện MSE so với baseline: {improvement:+.2f} percent")
