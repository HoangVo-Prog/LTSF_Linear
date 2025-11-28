# tuning.py

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.multioutput import MultiOutputRegressor

from config import HORIZON, RANDOM_STATE
from models_direct import Direct100Model
from models_multi import MultiStepModel
from evaluation_direct import evaluate_direct_model_on_fold
from evaluation_multi import evaluate_multi_model_on_fold


# ==========================
# 0. Hyperparameter grids
# ==========================

def generate_elasticnet_grid(
    alphas: List[float] = None,
    l1_ratios: List[float] = None,
) -> List[Dict[str, Any]]:
    if alphas is None:
        alphas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9]

    grid = []
    for a in alphas:
        for r in l1_ratios:
            grid.append({"alpha": a, "l1_ratio": r})
    return grid


def generate_ridge_grid(
    alphas: List[float] = None,
) -> List[Dict[str, Any]]:
    if alphas is None:
        alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    return [{"alpha": a} for a in alphas]


# ====================================
# 1. Pipeline 1 Direct scalar target
# ====================================

def tune_direct_elasticnet(
    df_direct: pd.DataFrame,
    folds: List[Dict],
    feature_cols: List[str],
    horizon: int = HORIZON,
    grid: List[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], float]:
    """
    Tuning ElasticNet cho Pipeline 1 (direct 100d).

    Output:
      - best_config
      - best_score (mean CV endpoint MSE)
    """
    if grid is None:
        grid = generate_elasticnet_grid()

    best_config: Dict[str, Any] = None
    best_score = np.inf

    for cfg in grid:
        cv_scores = []

        for fold in folds:
            base_model = ElasticNet(
                alpha=cfg["alpha"],
                l1_ratio=cfg["l1_ratio"],
                random_state=RANDOM_STATE,
                max_iter=5000,
            )
            model = Direct100Model(base_model)

            score = evaluate_direct_model_on_fold(
                df=df_direct,
                model=model,
                fold=fold,
                feature_cols=feature_cols,
                horizon=horizon,
            )
            cv_scores.append(score)

        mean_score = float(np.mean(cv_scores))
        print(f"[Direct ElasticNet] cfg={cfg} mean_MSE={mean_score:.6f}")

        if mean_score < best_score:
            best_score = mean_score
            best_config = cfg

    print("Best Direct ElasticNet config:", best_config, "score:", best_score)
    return best_config, best_score


def tune_direct_ridge(
    df_direct: pd.DataFrame,
    folds: List[Dict],
    feature_cols: List[str],
    horizon: int = HORIZON,
    grid: List[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], float]:
    """
    Tuning Ridge cho Pipeline 1 (direct 100d).
    """
    if grid is None:
        grid = generate_ridge_grid()

    best_config: Dict[str, Any] = None
    best_score = np.inf

    for cfg in grid:
        cv_scores = []

        for fold in folds:
            base_model = Ridge(
                alpha=cfg["alpha"],
                random_state=RANDOM_STATE,
            )
            # Dùng Direct100Model để giữ interface
            model = Direct100Model(base_model)

            score = evaluate_direct_model_on_fold(
                df=df_direct,
                model=model,
                fold=fold,
                feature_cols=feature_cols,
                horizon=horizon,
            )
            cv_scores.append(score)

        mean_score = float(np.mean(cv_scores))
        print(f"[Direct Ridge] cfg={cfg} mean_MSE={mean_score:.6f}")

        if mean_score < best_score:
            best_score = mean_score
            best_config = cfg

    print("Best Direct Ridge config:", best_config, "score:", best_score)
    return best_config, best_score


# ====================================
# 2. Pipeline 2 Multi step target
# ====================================

def tune_multi_elasticnet(
    df_multi: pd.DataFrame,
    Y_multi: np.ndarray,
    folds: List[Dict],
    feature_cols: List[str],
    horizon: int = HORIZON,
    grid: List[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], float]:
    """
    Tuning MultiOutput ElasticNet cho Pipeline 2 (multi step 100d).

    df_multi:
      - DataFrame có lp, time, feature_cols, index align với Y_multi
    Y_multi:
      - numpy array (N, H)
    """
    if grid is None:
        grid = generate_elasticnet_grid()

    best_config: Dict[str, Any] = None
    best_score = np.inf

    for cfg in grid:
        cv_scores = []

        for fold in folds:
            base = ElasticNet(
                alpha=cfg["alpha"],
                l1_ratio=cfg["l1_ratio"],
                random_state=RANDOM_STATE,
                max_iter=5000,
            )
            base_model = MultiOutputRegressor(base)
            model = MultiStepModel(base_model)

            score = evaluate_multi_model_on_fold(
                df=df_multi,
                Y_multi=Y_multi,
                model=model,
                fold=fold,
                feature_cols=feature_cols,
                horizon=horizon,
            )
            cv_scores.append(score)

        mean_score = float(np.mean(cv_scores))
        print(f"[Multi ElasticNet] cfg={cfg} mean_MSE100={mean_score:.6f}")

        if mean_score < best_score:
            best_score = mean_score
            best_config = cfg

    print("Best Multi ElasticNet config:", best_config, "score:", best_score)
    return best_config, best_score


def tune_multi_ridge(
    df_multi: pd.DataFrame,
    Y_multi: np.ndarray,
    folds: List[Dict],
    feature_cols: List[str],
    horizon: int = HORIZON,
    grid: List[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], float]:
    """
    Tuning MultiOutput Ridge cho Pipeline 2.
    """
    if grid is None:
        grid = generate_ridge_grid()

    best_config: Dict[str, Any] = None
    best_score = np.inf

    for cfg in grid:
        cv_scores = []

        for fold in folds:
            base = Ridge(
                alpha=cfg["alpha"],
                random_state=RANDOM_STATE,
            )
            base_model = MultiOutputRegressor(base)
            model = MultiStepModel(base_model)

            score = evaluate_multi_model_on_fold(
                df=df_multi,
                Y_multi=Y_multi,
                model=model,
                fold=fold,
                feature_cols=feature_cols,
                horizon=horizon,
            )
            cv_scores.append(score)

        mean_score = float(np.mean(cv_scores))
        print(f"[Multi Ridge] cfg={cfg} mean_MSE100={mean_score:.6f}")

        if mean_score < best_score:
            best_score = mean_score
            best_config = cfg

    print("Best Multi Ridge config:", best_config, "score:", best_score)
    return best_config, best_score
