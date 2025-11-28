# optimize_residual.py

import numpy as np
from typing import Dict, Any, Tuple
from residual_model import ResidualModel


def random_search_elasticnet(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials: int = 25,
    random_state: int = 42,
) -> Tuple[ResidualModel, Dict[str, Any]]:
    rng = np.random.default_rng(random_state)
    best_score = float("inf")
    best_params: Dict[str, Any] = {}
    best_model: ResidualModel = None

    for _ in range(n_trials):
        alpha = 10 ** rng.uniform(-4, 0)         # 0.0001 to 1
        l1_ratio = rng.uniform(0.0, 1.0)         # 0 to 1

        model = ResidualModel(
            model_type="elasticnet",
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=10000,
        )
        model.fit(X_train, y_train)
        mse_val = model.score_mse(X_val, y_val)

        if mse_val < best_score:
            best_score = mse_val
            best_model = model
            best_params = {"alpha": alpha, "l1_ratio": l1_ratio}

    return best_model, best_params


def random_search_ridge(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials: int = 25,
    random_state: int = 42,
) -> Tuple[ResidualModel, Dict[str, Any]]:
    rng = np.random.default_rng(random_state)
    best_score = float("inf")
    best_params: Dict[str, Any] = {}
    best_model: ResidualModel = None

    for _ in range(n_trials):
        alpha = 10 ** rng.uniform(-4, 2)  # 0.0001 to 100

        model = ResidualModel(
            model_type="ridge",
            alpha=alpha,
        )
        model.fit(X_train, y_train)
        mse_val = model.score_mse(X_val, y_val)

        if mse_val < best_score:
            best_score = mse_val
            best_model = model
            best_params = {"alpha": alpha}

    return best_model, best_params
