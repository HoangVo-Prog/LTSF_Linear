# feature_selection.py

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


from p1.evaluation_direct import compute_endpoint_price_from_direct


from config import TOP_K_FEATURES, RANDOM_STATE

try:
    import xgboost as xgb
except ImportError:
    xgb = None


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


def _normalize_importance(
    imp: pd.Series,
    clip_negative: bool = True,
) -> pd.Series:
    """
    Chuẩn hóa importance về [0,1] với max = 1.
    Nếu clip_negative True, các giá trị âm bị đặt về 0.
    """
    s = imp.copy()
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clip_negative:
        s = s.clip(lower=0.0)

    max_val = s.abs().max()
    if max_val <= 0:
        return s * 0.0
    return s / max_val


def compute_feature_importances_all_models_per_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    base_model_enet: ElasticNet = None,
) -> Dict[str, pd.Series]:
    """
    Tính importance từ nhiều baseline model trên một fold:
      - ElasticNet + permutation importance
      - RandomForestRegressor feature_importances_
      - XGBRegressor feature_importances_ (nếu có)

    Trả về dict: { "enet": series, "rf": series, "xgb": series(optional) }
    """
    importances: Dict[str, pd.Series] = {}

    # 1. ElasticNet + permutation importance
    if base_model_enet is None:
        enet = ElasticNet(
            alpha=1e-3,
            l1_ratio=0.5,
            random_state=RANDOM_STATE,
            max_iter=2000,
        )
    else:
        enet = ElasticNet(
            alpha=base_model_enet.alpha,
            l1_ratio=base_model_enet.l1_ratio,
            random_state=RANDOM_STATE,
            max_iter=base_model_enet.max_iter,
        )

    enet.fit(X_train, y_train)
    result = permutation_importance(
        enet,
        X_val,
        y_val,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="neg_mean_squared_error",
    )
    imp_enet = pd.Series(result.importances_mean, index=X_val.columns)
    importances["enet"] = _normalize_importance(imp_enet, clip_negative=True)

    # 2. RandomForestRegressor
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    imp_rf = pd.Series(rf.feature_importances_, index=X_train.columns)
    importances["rf"] = _normalize_importance(imp_rf, clip_negative=False)

    # 3. XGBoost nếu có
    if xgb is not None:
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1,
        )
        xgb_model.fit(X_train.values, y_train)
        imp_xgb = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
        importances["xgb"] = _normalize_importance(imp_xgb, clip_negative=False)

    return importances


def aggregate_feature_importances_multi_model(
    per_fold_importances: Dict[str, List[pd.Series]],
    min_folds_used: int = 1,
    top_k: int = TOP_K_FEATURES,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Gộp importance từ nhiều baseline model (ElasticNet, RF, XGB) giữa các fold.

    per_fold_importances:
      dict model_name -> list[pd.Series] (mỗi series là 1 fold)

    Output:
      - selected_features: list tên feature
      - rank_df: DataFrame chi tiết với:
          enet_importance_mean, rf_importance_mean, xgb_importance_mean,
          rank_score (trung bình các model)
    """
    model_names = list(per_fold_importances.keys())
    if len(model_names) == 0:
        raise ValueError("No model importances passed to aggregate_feature_importances_multi_model")

    # union tất cả feature
    all_features = sorted(
        {
            f
            for m in model_names
            for s in per_fold_importances[m]
            for f in s.index
        }
    )

    # lưu thống kê theo từng model
    stats = {}
    for m in model_names:
        folds_series = per_fold_importances[m]
        n_folds_m = len(folds_series)

        imp_mat = pd.DataFrame(index=all_features, columns=range(n_folds_m), dtype=float)
        for i, s in enumerate(folds_series):
            imp_mat[i] = s.reindex(all_features)

        imp_mean = imp_mat.mean(axis=1, skipna=True)
        folds_used = imp_mat.notna().sum(axis=1)

        rank_score_m = imp_mean * (folds_used / max(1, n_folds_m))

        stats[m] = {
            "importance_mean": imp_mean,
            "folds_used": folds_used,
            "rank_score": rank_score_m,
        }

    # build rank_df tổng hợp
    rank_df = pd.DataFrame(index=all_features)

    rank_score_list = []
    for m in model_names:
        rank_df[f"{m}_importance_mean"] = stats[m]["importance_mean"]
        rank_df[f"{m}_folds_used"] = stats[m]["folds_used"]
        rank_df[f"{m}_rank_score"] = stats[m]["rank_score"]
        rank_score_list.append(stats[m]["rank_score"])

    # final rank_score = mean rank_score của các model
    rank_df["rank_score"] = pd.concat(rank_score_list, axis=1).mean(axis=1)

    # tổng folds_used = max số fold mà feature xuất hiện ở bất kỳ model nào
    folds_used_total = None
    for m in model_names:
        fu = stats[m]["folds_used"]
        folds_used_total = fu if folds_used_total is None else fu.combine(folds_used_total, func=np.maximum)
    rank_df["folds_used_total"] = folds_used_total

    # lọc theo min_folds_used
    rank_df = rank_df[rank_df["folds_used_total"] >= min_folds_used]

    # sort theo rank_score giảm dần
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

    Bước 1: dùng ENet + RF + XGB trên endpoint price để lấy importance per-fold.
    Bước 2: gộp lại qua aggregate_feature_importances_multi_model -> rank_score.
    Bước 3: kết hợp thêm |corr(feature, endpoint_price)| để có combined_score.
    """

    per_fold_importances: Dict[str, List[pd.Series]] = {}

    # ===== Bước 1: importance theo model trên từng fold =====
    for fold in folds:
        train_mask = fold["train_mask"]
        val_mask = fold["val_mask"]

        df_train = df_direct.loc[train_mask]
        df_val = df_direct.loc[val_mask]

        X_train = df_train[feature_cols]
        # y_train_price: endpoint price từ y_direct của train
        # dùng cùng công thức lp + y_direct như tuning
        lp_train = df_train["lp"].values
        y_train_direct = df_train["y_direct"].values
        y_train_price = np.exp(lp_train + y_train_direct)

        X_val = df_val[feature_cols]
        lp_val = df_val["lp"].values
        y_val_direct = df_val["y_direct"].values
        y_val_price = np.exp(lp_val + y_val_direct)

        imp_dict = compute_feature_importances_all_models_per_fold(
            X_train,
            y_train_price,
            X_val,
            y_val_price,
            base_model_enet=base_model_for_importance,
        )

        for m_name, imp_series in imp_dict.items():
            per_fold_importances.setdefault(m_name, []).append(imp_series)

    # ===== Bước 2: gộp importance giữa các fold / model =====
    # rank_score ở đây là "score từ model" (ENet/RF/XGB)
    selected_features_tmp, rank_df = aggregate_feature_importances_multi_model(
        per_fold_importances=per_fold_importances,
        min_folds_used=min_folds_used,
        top_k=top_k,
    )

    # ===== Bước 3: thêm thông tin corr với endpoint price toàn sample =====
    lp_all = df_direct["lp"].values
    y_direct_all = df_direct["y_direct"].values
    y_price_all = np.exp(lp_all + y_direct_all)

    corr_abs_list = []
    for feat in rank_df.index:
        x = df_direct[feat].values

        mask = np.isfinite(x) & np.isfinite(y_price_all)
        if mask.sum() < 30:
            # quá ít điểm hữu ích -> coi như không đáng tin
            corr = 0.0
        else:
            x_centered = x[mask] - x[mask].mean()
            y_centered = y_price_all[mask] - y_price_all[mask].mean()
            denom = np.sqrt((x_centered ** 2).sum()) * np.sqrt((y_centered ** 2).sum())
            if denom <= 0:
                corr = 0.0
            else:
                corr = float((x_centered * y_centered).sum() / denom)

        corr_abs_list.append(abs(corr))

    rank_df["corr_abs"] = corr_abs_list

    # Chuẩn hóa rank_score và corr_abs về [0,1] để cộng được
    rs = rank_df["rank_score"].values
    max_rs = np.nanmax(np.abs(rs)) if len(rs) > 0 else 0.0
    if max_rs > 0:
        rank_df["rank_score_norm"] = rank_df["rank_score"] / max_rs
    else:
        rank_df["rank_score_norm"] = 0.0

    max_corr = rank_df["corr_abs"].max() if len(rank_df) > 0 else 0.0
    if max_corr > 0:
        rank_df["corr_abs_norm"] = rank_df["corr_abs"] / max_corr
    else:
        rank_df["corr_abs_norm"] = 0.0

    # Kết hợp: có thể điều chỉnh alpha nếu muốn nghiêng về model/corr hơn
    alpha = 0.5
    rank_df["combined_score"] = (
        alpha * rank_df["rank_score_norm"] + (1.0 - alpha) * rank_df["corr_abs_norm"]
    )

    # Sort lại theo combined_score và lấy top_k
    rank_df = rank_df.sort_values("combined_score", ascending=False)
    selected_features = rank_df.index.tolist()[:top_k]

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

    per_fold_importances: Dict[str, List[pd.Series]] = {}

    for fold in folds:
        train_mask = fold["train_mask"]
        val_mask = fold["val_mask"]

        train_idx = df_multi.index[train_mask]
        val_idx = df_multi.index[val_mask]

        X_train = df_multi.loc[train_idx, feature_cols]
        y_train = y_scalar_all[train_idx]

        X_val = df_multi.loc[val_idx, feature_cols]
        y_val = y_scalar_all[val_idx]

        imp_dict = compute_feature_importances_all_models_per_fold(
            X_train,
            y_train,
            X_val,
            y_val,
            base_model_enet=base_model_for_importance,
        )

        for m_name, imp_series in imp_dict.items():
            per_fold_importances.setdefault(m_name, []).append(imp_series)

    selected_features, rank_df = aggregate_feature_importances_multi_model(
        per_fold_importances=per_fold_importances,
        min_folds_used=min_folds_used,
        top_k=top_k,
    )

    return selected_features, rank_df
