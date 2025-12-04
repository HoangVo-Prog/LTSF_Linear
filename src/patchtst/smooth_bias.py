# patchtst/smooth_bias.py

from typing import Tuple

import numpy as np

from metrics import compute_metrics


def smooth_bias_correction_with_postprocessing(
    pred_baseline: np.ndarray,
    post_model,
    method: str = "linear",
    smooth_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Áp dụng smooth bias correction kết hợp với post processing regression.
    - Giai đoạn đầu: smooth transition baseline -> post
    - Giai đoạn cuối: dùng post processing hoàn toàn
    """

    n = len(pred_baseline)

    pred_post = post_model.predict(pred_baseline.reshape(-1, 1))

    weights = np.zeros(n, dtype=float)

    split_point = int(n * smooth_ratio)
    if split_point < 1:
        split_point = 1
    if split_point >= n - 1:
        split_point = max(1, n - 2)

    if split_point > 1:
        smooth_part = split_point
        if method == "linear":
            smooth_weights = np.arange(smooth_part) / max(smooth_part - 1, 1)
        else:
            smooth_weights = np.arange(smooth_part) / max(smooth_part - 1, 1)

        if smooth_weights[-1] > 0:
            smooth_weights = smooth_weights / smooth_weights[-1]
        weights[:split_point] = smooth_weights

        weights[0] = 0.0
        weights[split_point - 1] = 1.0

    weights[split_point:] = 1.0
    weights[0] = 0.0
    weights[-1] = 1.0

    pred_corrected = (1.0 - weights) * pred_baseline + weights * pred_post

    if split_point < len(pred_corrected):
        pred_corrected[split_point:] = pred_post[split_point:]
        weights[split_point:] = 1.0

    return pred_corrected, weights


def evaluate_smooth_method(
    pred_baseline: np.ndarray,
    pred_smooth: np.ndarray,
    post_model,
    y_true,
    baseline_mse: float,
    post_mse: float,
):
    """In metrics cho phương pháp smooth linear best method."""
    metrics_smooth = compute_metrics(y_true, pred_smooth)

    print("\n Smooth Linear 20 percent Results (BEST METHOD):")
    print(f" - MSE:  {metrics_smooth['mse']:.4f} (Baseline: {baseline_mse:.4f}, Post: {post_mse:.4f})")
    print(f" - RMSE: {metrics_smooth['rmse']:.4f}")
    print(f" - MAE:  {metrics_smooth['mae']:.4f}")
    print(f" - R2:   {metrics_smooth['r2']:.4f}")
    print(f" - MAPE: {metrics_smooth['mape']:.2f} percent")
    print(f" - Bias: {metrics_smooth['bias']:.4f}")

    pred_post_best = post_model.predict(pred_baseline.reshape(-1, 1))

    first_value_preserved = np.isclose(
        pred_smooth[0], pred_baseline[0], rtol=1e-5
    )
    last_value_matches_post = np.isclose(
        pred_smooth[-1], pred_post_best[-1], rtol=1e-5
    )

    print(
        f" - Giá trị đầu: {pred_smooth[0]:.4f} "
        f"(Baseline: {pred_baseline[0]:.4f}, Giữ nguyên: {bool(first_value_preserved)})"
    )
    print(
        f" - Giá trị cuối: {pred_smooth[-1]:.4f} "
        f"(Post: {pred_post_best[-1]:.4f}, Khớp: {bool(last_value_matches_post)})"
    )

    improvement_vs_baseline = (baseline_mse - metrics_smooth["mse"]) / baseline_mse * 100.0
    improvement_vs_post = (post_mse - metrics_smooth["mse"]) / post_mse * 100.0

    print("\n Cải thiện:")
    print(f" - So với baseline:      {improvement_vs_baseline:+.2f} percent")
    print(f" - So với post processing: {improvement_vs_post:+.2f} percent")

    return metrics_smooth
