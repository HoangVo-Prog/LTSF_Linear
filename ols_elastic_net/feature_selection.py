# feature_selection.py

from typing import List
import numpy as np
from sklearn.inspection import permutation_importance
from residual_model import ResidualModel


def compute_feature_importance(
    model: ResidualModel,
    X_val,
    y_val,
    n_repeats: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    result = permutation_importance(
        model.model, X_val, y_val, n_repeats=n_repeats, random_state=random_state
    )
    # shape: (n_features,)
    return result.importances_mean


def select_top_k_features(
    importances: np.ndarray,
    feature_names: List[str],
    k: int,
) -> List[str]:
    if len(importances) != len(feature_names):
        raise ValueError(
            f"importances length {len(importances)} "
            f"!= feature_names length {len(feature_names)}"
        )

    idx_sorted = np.argsort(importances)[::-1]
    k = min(k, len(idx_sorted))
    top_idx = idx_sorted[:k]

    return [feature_names[i] for i in top_idx]


def filter_feature_matrix(X, selected_features: List[str]):
    return X.loc[:, selected_features]
