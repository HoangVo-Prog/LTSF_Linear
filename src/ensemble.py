from typing import List, Dict
import numpy as np


def simple_weight_search(
    price_hat_paths_list: List[np.ndarray],
    price_true_paths: np.ndarray,
    step: float = 0.05,
) -> np.ndarray:
    """
    Ensemble đơn giản trên simplex khi có ít model.
    price_hat_paths_list: list length M, mỗi phần shape (N_eval, H)
    price_true_paths: shape (N_eval, H)
    Trả về: vector weight dài M.
    Ở đây chỉ demo 2 model, nhiều model bạn chuyển sang random search.
    """
    M = len(price_hat_paths_list)
    if M == 1:
        return np.array([1.0])

    # Demo chỉ cho 2 model
    best_w = np.array([0.5, 0.5])
    best_score = np.inf

    for w0 in np.arange(0.0, 1.0 + 1e-9, step):
        w = np.array([w0, 1.0 - w0])
        ensemble_paths = w[0] * price_hat_paths_list[0] + w[1] * price_hat_paths_list[1]
        diff = ensemble_paths - price_true_paths
        score = float(np.mean(diff ** 2))
        if score < best_score:
            best_score = score
            best_w = w

    return best_w
