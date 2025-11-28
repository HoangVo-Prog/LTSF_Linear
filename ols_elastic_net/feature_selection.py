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
    return result.importances_mean


def select_top_k_features(
    importances: np.ndarray,
    feature_names: List[str],
    k: int,
) -> List[str]:
    # Defensive: align lengths
    n_imp = len(importances)
    n_feat = len(feature_names)
    n = min(k, n_imp, n_feat)

    if n_imp != n_feat:
        print(
            f"[Warning] importances length {n_imp} != feature_names length {n_feat}. "
            f"Using first {n} features to align."
        )

    # Sort indices by importance descending
    idx_sorted = np.argsort(importances)[::-1]

    # Clip to available features
    top_idx = idx_sorted[:n]

    # Also clip feature_names if lengths differ
    safe_feature_names = feature_names[:n_imp]

    return [safe_feature_names[i] for i in top_idx]


def filter_feature_matrix(X, selected_features: List[str]):
    return X.loc[:, selected_features]
