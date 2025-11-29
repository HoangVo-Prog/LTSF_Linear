# tuning_direct.py

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


from models_direct import MODEL_REGISTRY, Direct100Model

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HORIZON, RANDOM_STATE
from evaluation_direct import compute_endpoint_price_from_direct, mse


def build_grid_for_model(model_name: str) -> List[Dict[str, Any]]:
    """
    Sinh grid khá sâu cho từng model type.
    Nếu quá nặng có thể cắt bớt bằng max_configs khi gọi tune_model_direct.
    """
    # 1. ElasticNet: nhiều alpha + l1_ratio + scale_target
    if model_name == "elasticnet":
        alphas = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        scale_targets = [True, False]
        grid = []
        for a in alphas:
            for r in l1_ratios:
                for st in scale_targets:
                    grid.append(
                        {
                            "alpha": a,
                            "l1_ratio": r,
                            "max_iter": 5000,
                            "scale_target": st,
                        }
                    )
        return grid

    # 2. Ridge: alpha logspace + scale_target
    if model_name == "ridge":
        alphas = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1, 1.0]
        scale_targets = [True, False]
        grid = []
        for a in alphas:
            for st in scale_targets:
                grid.append({"alpha": a, "scale_target": st})
        return grid

    # 3. XGBoost: sâu hơn nhưng vẫn vừa phải
    if model_name == "xgboost":
        grid = []
        for n in [400, 800]:
            for depth in [3, 5]:
                for lr in [0.02, 0.03, 0.05]:
                    for subs in [0.8, 1.0]:
                        # đơn giản hóa colsample/reg cho đỡ nổ grid
                        grid.append(
                            {
                                "n_estimators": n,
                                "max_depth": depth,
                                "learning_rate": lr,
                                "subsample": subs,
                                "colsample_bytree": 0.8,
                                "tree_method": "hist",
                                "reg_lambda": 1.0,
                                "reg_alpha": 0.0,
                            }
                        )
        return grid

    # 4. LightGBM: thêm depth, lr, subsample, min_child_samples
    if model_name == "lgbm":
        grid = []
        for n in [400, 800]:
            for depth in [-1, 4, 6]:
                for lr in [0.02, 0.03, 0.05]:
                    for subs in [0.8, 1.0]:
                        for mcs in [20, 40]:
                            grid.append(
                                {
                                    "n_estimators": n,
                                    "max_depth": depth,
                                    "learning_rate": lr,
                                    "subsample": subs,
                                    "colsample_bytree": 0.8,
                                    "min_child_samples": mcs,
                                }
                            )
        return grid

    # 5. RandomForest: thêm n_estimators, depth, min_samples_leaf, max_features
    if model_name == "random_forest":
        grid = []
        for n in [200, 400, 800]:
            for depth in [None, 6, 10]:
                for leaf in [1, 2, 4]:
                    for mf in ["sqrt", "log2"]:
                        grid.append(
                            {
                                "n_estimators": n,
                                "max_depth": depth,
                                "min_samples_leaf": leaf,
                                "max_features": mf,
                            }
                        )
        return grid

    # 6. GradientBoosting (GBDT): thêm n, depth, lr, subsample
    if model_name == "gbdt":
        grid = []
        for n in [200, 400, 800]:
            for depth in [2, 3, 4]:
                for lr in [0.02, 0.03, 0.05]:
                    for subs in [0.8, 1.0]:
                        grid.append(
                            {
                                "n_estimators": n,
                                "max_depth": depth,
                                "learning_rate": lr,
                                "subsample": subs,
                            }
                        )
        return grid

    # 7. DLinear: chơi với scale_target + fit_intercept
    if model_name == "dlinear":
        grid = []
        for st in [True, False]:
            for fi in [True, False]:
                grid.append(
                    {
                        "scale_target": st,
                        "fit_intercept": fi,
                    }
                )
        return grid

    # 8. NLinear: tương tự nhưng mặc định hay dùng không intercept
    if model_name == "nlinear":
        grid = []
        for st in [True, False]:
            for fi in [False, True]:
                grid.append(
                    {
                        "scale_target": st,
                        "fit_intercept": fi,
                    }
                )
        return grid

    # Rolling, Kalman: tạm thời chưa có grid
    return [{}]


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def evaluate_model_one_fold_direct(
    df_direct: pd.DataFrame,
    model: Direct100Model,
    fold: Dict,
    feature_cols: List[str],
    horizon: int = HORIZON,
) -> float:
    train_mask = fold["train_mask"]
    val_mask = fold["val_mask"]

    df_train = df_direct.loc[train_mask]
    df_val = df_direct.loc[val_mask]

    if df_val.empty:
        return np.inf

    X_train = df_train[feature_cols]
    y_train = df_train["y_direct"].values

    X_val = df_val[feature_cols]
    # y_true = df_val["y_direct"].values  # không cần trực tiếp nữa

    model.fit(X_train, y_train)
    y_hat = model.predict_100day_return(X_val)

    price_true, price_hat = compute_endpoint_price_from_direct(df_val, y_hat)
    return mse(price_true, price_hat)


def tune_model_direct(
    model_name: str,
    df_direct: pd.DataFrame,
    folds: List[Dict],
    feature_cols: List[str],
    horizon: int = HORIZON,
    max_configs: int = None,
) -> Tuple[Dict[str, Any], float]:
    """
    Hyperparameter tuning cho 1 model type.

    Trả về:
      best_config, best_score (mean CV)
    """
    ModelClass = MODEL_REGISTRY[model_name]
    grid = build_grid_for_model(model_name)

    if max_configs is not None:
        grid = grid[:max_configs]

    best_config = None
    best_score = np.inf

    for cfg in grid:
        fold_scores = []
        for fold in folds:
            model = ModelClass(cfg)
            score = evaluate_model_one_fold_direct(
                df_direct=df_direct,
                model=model,
                fold=fold,
                feature_cols=feature_cols,
                horizon=horizon,
            )
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        print(f"[Direct {model_name}] cfg={cfg} mean_MSE={mean_score:.6f}")

        if mean_score < best_score:
            best_score = mean_score
            best_config = cfg

    print(f"[Direct {model_name}] best_config={best_config}, best_score={best_score:.6f}")
    return best_config, best_score


def collect_validation_predictions_direct(
    model_name: str,
    best_config: Dict[str, Any],
    df_direct: pd.DataFrame,
    folds: List[Dict],
    feature_cols: List[str],
    horizon: int = HORIZON,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chạy lại CV với best_config để collect:
      - price_true_all: shape (N,)
      - price_hat_all: shape (N,)
    (endpoint price tại t+H)
    """
    ModelClass = MODEL_REGISTRY[model_name]

    all_true = []
    all_hat = []

    for fold in folds:
        train_mask = fold["train_mask"]
        val_mask = fold["val_mask"]

        df_train = df_direct.loc[train_mask]
        df_val = df_direct.loc[val_mask]

        if df_val.empty:
            continue

        X_train = df_train[feature_cols]
        y_train = df_train["y_direct"].values

        X_val = df_val[feature_cols]
        y_val = df_val["y_direct"].values

        model = ModelClass(best_config)
        model.fit(X_train, y_train)
        y_hat = model.predict_100day_return(X_val)

        # convert sang price
        lp_val = df_val["lp"].values
        price_true = np.exp(lp_val + y_val)
        price_hat = np.exp(lp_val + y_hat)

        all_true.append(price_true)
        all_hat.append(price_hat)

    if not all_true:
        raise RuntimeError("No validation samples collected for ensemble")

    price_true_all = np.concatenate(all_true)
    price_hat_all = np.concatenate(all_hat)
    return price_true_all, price_hat_all
