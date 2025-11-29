# ensemble.py

from typing import List, Dict, Tuple
import numpy as np

from sklearn.linear_model import Ridge, ElasticNet


def compute_price_endpoint_from_R(
    lp: np.ndarray,
    start_indices: np.ndarray,
    R_hat: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Từ R_hat (log return 100 ngày) -> price_hat_T, price_true_T.
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
    Giữ lại hàm random search cũ nếu muốn so sánh.
    """
    N, M = price_hat_matrix.shape
    if M == 1:
        return np.array([1.0], dtype=float)

    w_equal = np.ones(M) / M
    best_w = w_equal.copy()
    best_score = np.inf

    rng = np.random.default_rng(42)

    for _ in range(n_samples):
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
    Giữ lại nếu vẫn muốn dùng random search + shrink.
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


# ============================================================
# Stacking meta learner trên OOF prediction
# ============================================================

def train_stacking_meta_learner(
    oof_pred_matrix: np.ndarray,
    y_true: np.ndarray,
    model_type: str = "ridge",
    positive: bool = False,
) -> object:
    """
    Train meta learner trên OOF prediction.

    oof_pred_matrix: shape (N, M)
      N: số điểm validation (gộp tất cả fold)
      M: số model base
    y_true: shape (N,)
      y_direct thực tế (log return 100 ngày)

    model_type:
      - "ridge": RidgeRegression
      - "elasticnet": ElasticNet

    positive:
      - Nếu True thì ép weight không âm (chỉ áp dụng cho Ridge).
    """
    X = np.asarray(oof_pred_matrix, dtype=float)
    y = np.asarray(y_true, dtype=float)

    if model_type == "ridge":
        # alpha cố định hoặc bạn có thể cho một grid nhỏ
        meta = Ridge(
            alpha=1e-2,
            fit_intercept=True,
            random_state=42,
            positive=positive,
        )
    elif model_type == "elasticnet":
        meta = ElasticNet(
            alpha=1e-3,
            l1_ratio=0.5,
            max_iter=5000,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown meta model_type: {model_type}")

    meta.fit(X, y)
    return meta


def predict_with_meta_learner(
    meta_model: object,
    base_preds: np.ndarray,
) -> np.ndarray:
    """
    Dự báo bằng meta learner.

    base_preds: shape (N, M) hoặc (M,)
      - Nếu là (M,) sẽ reshape thành (1, M).
    """
    arr = np.asarray(base_preds, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return meta_model.predict(arr)
