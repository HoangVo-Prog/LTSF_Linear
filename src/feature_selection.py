# feature_selection.py

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.inspection import permutation_importance

from config import TOP_K_FEATURES, RANDOM_STATE


def build_selection_target_from_multi(Y_multi: np.ndarray, mode: str = "sum") -> np.ndarray:
    """
    Chuyển target multi output (N, H) thành scalar cho bước feature selection.

    mode:
      - "sum": tổng return 100 ngày
      - "mean": trung bình return 100 ngày
      - "endpoint": dùng return endpoint lp_{t+H} - lp_t
    """
    if mode == "sum":
        return Y_multi.sum(axis=1)
    elif mode == "mean":
        return Y_multi.mean(axis=1)
    elif mode == "endpoint":
        return Y_multi.sum(axis=1)
    else:
        raise ValueError(f"Unknown mode for selection target: {mode}")


def compute_feature_importance_per_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    base_model: ElasticNet = None,
) -> pd.Series:
    """
    Train baseline ElasticNet và compute permutation importance trên X_val.

    base_model:
      - nếu None sẽ dùng ElasticNet với default nhẹ.
      - nếu không, dùng bản copy cấu hình bạn truyền vào.
    """
    if base_model is None:
        model = ElasticNet(
            alpha=1e-3,
            l1_ratio=0.5,
            random_state=RANDOM_STATE,
            max_iter=2000,
        )
    else:
        # clone đơn giản
        model = ElasticNet(
            alpha=base_model.alpha,
            l1_ratio=base_model.l1_ratio,
            random_state=RANDOM_STATE,
            max_iter=base_model.max_iter,
        )

    model.fit(X_train, y_train)

    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="neg_mean_squared_error",
    )
    importances = pd.Series(result.importances_mean, index=X_val.columns)
    return importances


def aggregate_feature_importances(
    per_fold_importances: List[pd.Series],
    min_folds_used: int = 1,
    top_k: int = TOP_K_FEATURES,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Gộp importance giữa các fold, chọn top_k feature và trả về
    cả bảng ranking chi tiết.

    Output:
      - selected_features: list tên cột
      - rank_df: DataFrame có cột:
          importance_mean, folds_used, rank_score
    """
    all_features = sorted({f for s in per_fold_importances for f in s.index})
    n_folds = len(per_fold_importances)

    imp_mat = pd.DataFrame(index=all_features, columns=range(n_folds), dtype=float)
    for i, s in enumerate(per_fold_importances):
        imp_mat[i] = s.reindex(all_features)

    imp_mean = imp_mat.mean(axis=1, skipna=True)
    folds_used = imp_mat.notna().sum(axis=1)

    rank_score = imp_mean * (folds_used / n_folds)
    rank_df = pd.DataFrame(
        {
            "importance_mean": imp_mean,
            "folds_used": folds_used,
            "rank_score": rank_score,
        }
    )

    rank_df = rank_df[rank_df["folds_used"] >= min_folds_used]
    rank_df = rank_df.sort_values("rank_score", ascending=False)

    selected_features = rank_df.index.tolist()[:top_k]
    return selected_features, rank_df


# ==========================
# 1. Direct pipeline (scalar)
# ==========================

def run_feature_selection_direct(
    df_direct: pd.DataFrame,
    folds: List[Dict],
    feature_cols: List[str],
    top_k: int = TOP_K_FEATURES,
    base_model_for_importance: ElasticNet = None,
    min_folds_used: int = 1,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Feature selection cho Pipeline 1 Direct 100d.

    df_direct:
      - Đã build features, đã có y_direct
    folds:
      - output của make_folds trên df_direct["time"]
    feature_cols:
      - list tên feature ban đầu
    """
    per_fold_importances: List[pd.Series] = []

    for i, fold in enumerate(folds):
        train_mask = fold["train_mask"]
        val_mask = fold["val_mask"]

        df_train = df_direct.loc[train_mask]
        df_val = df_direct.loc[val_mask]

        X_train = df_train[feature_cols]
        y_train = df_train["y_direct"].values

        X_val = df_val[feature_cols]
        y_val = df_val["y_direct"].values

        imp = compute_feature_importance_per_fold(
            X_train,
            y_train,
            X_val,
            y_val,
            base_model=base_model_for_importance,
        )
        per_fold_importances.append(imp)

    selected_features, rank_df = aggregate_feature_importances(
        per_fold_importances,
        min_folds_used=min_folds_used,
        top_k=top_k,
    )

    return selected_features, rank_df


# ==========================
# 2. Multi step pipeline
# ==========================

def run_feature_selection_multi(
    df_multi: pd.DataFrame,
    Y_multi: np.ndarray,
    folds: List[Dict],
    feature_cols: List[str],
    top_k: int = TOP_K_FEATURES,
    base_model_for_importance: ElasticNet = None,
    min_folds_used: int = 1,
    selection_target_mode: str = "sum",
) -> Tuple[List[str], pd.DataFrame]:
    """
    Feature selection cho Pipeline 2 Multi step.

    df_multi:
      - DataFrame đã build features, align index với Y_multi
    Y_multi:
      - numpy array shape (N, H)
    folds:
      - output của make_folds trên df_multi["time"]
    feature_cols:
      - list tên feature ban đầu
    """
    # build scalar target từ multi output
    y_scalar_all = build_selection_target_from_multi(Y_multi, mode=selection_target_mode)

    per_fold_importances: List[pd.Series] = []

    for i, fold in enumerate(folds):
        train_mask = fold["train_mask"]
        val_mask = fold["val_mask"]

        train_idx = df_multi.index[train_mask]
        val_idx = df_multi.index[val_mask]

        X_train = df_multi.loc[train_idx, feature_cols]
        y_train = y_scalar_all[train_idx]

        X_val = df_multi.loc[val_idx, feature_cols]
        y_val = y_scalar_all[val_idx]

        imp = compute_feature_importance_per_fold(
            X_train,
            y_train,
            X_val,
            y_val,
            base_model=base_model_for_importance,
        )
        per_fold_importances.append(imp)

    selected_features, rank_df = aggregate_feature_importances(
        per_fold_importances,
        min_folds_used=min_folds_used,
        top_k=top_k,
    )

    return selected_features, rank_df
