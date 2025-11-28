# ensemble.py

from typing import List, Dict, Tuple
import numpy as np


def compute_price_endpoint_from_R(
    lp: np.ndarray,
    start_indices: np.ndarray,
    R_hat: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Từ R_hat (log return 100 ngày) -> price_hat_T, price_true_T.

    lp: array log price
    start_indices: index t
    R_hat: array shape (N_eval,) tương ứng với t
    """
    price_hat_T = []
    price_true_T = []

    for i, t in enumerate(start_indices):
        lp_t = lp[t]
        lp_T_true = lp[t + horizon]
        price_true_T.append(np.exp(lp_T_true))

        lp_T_hat = lp_t + R_hat[i]
        price_hat_T.append(np.exp(lp_T_hat))

    return np.array(price_true_T), np.array(price_hat_T)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def tune_ensemble_weights_random_search(
    price_hat_matrix: np.ndarray,
    price_true: np.ndarray,
    n_samples: int = 5000,
    l2_shrink: float = 0.0,
) -> np.ndarray:
    """
    Tuning lần 1 cho ensemble: tìm weight tối ưu trên simplex.

    price_hat_matrix: shape (N, M)
      - N: số điểm validation (gom tất cả fold lại)
      - M: số model
    price_true: shape (N,)

    n_samples: số vector weight random trên simplex.
    l2_shrink: nếu > 0 thì thêm penalty lambda * ||w - w_equal||^2.

    Output:
      w_best: vector length M
    """
    N, M = price_hat_matrix.shape
    if M == 1:
        return np.array([1.0], dtype=float)

    w_equal = np.ones(M) / M
    best_w = w_equal.copy()
    best_score = np.inf

    rng = np.random.default_rng(42)

    for _ in range(n_samples):
        # random trên simplex
        raw = rng.random(M)
        w = raw / raw.sum()

        y_hat = price_hat_matrix @ w
        base_loss = mse(price_true, y_hat)
        if l2_shrink > 0:
            reg = l2_shrink * float(np.sum((w - w_equal) ** 2))
        else:
            reg = 0.0

        score = base_loss + reg
        if score < best_score:
            best_score = score
            best_w = w

    return best_w


def tune_ensemble_shrinkage(
    price_hat_matrix: np.ndarray,
    price_true: np.ndarray,
    w_star: np.ndarray,
    shrink_values: List[float] = None,
) -> Tuple[float, np.ndarray]:
    """
    Tuning lần 2: shrink w_star về w_equal.

    shrink_values: list lambda in [0,1].
      w(lambda) = lambda * w_star + (1 - lambda) * w_equal.

    Trả về:
      best_lambda, w_opt
    """
    N, M = price_hat_matrix.shape
    w_equal = np.ones(M) / M

    if shrink_values is None:
        shrink_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    best_lambda = 1.0
    best_w = w_star.copy()
    best_score = np.inf

    for lam in shrink_values:
        w = lam * w_star + (1.0 - lam) * w_equal
        y_hat = price_hat_matrix @ w
        score = mse(price_true, y_hat)
        if score < best_score:
            best_score = score
            best_lambda = lam
            best_w = w

    return best_lambda, best_w
