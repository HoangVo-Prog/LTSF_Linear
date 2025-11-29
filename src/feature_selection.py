from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.feature_selection import mutual_info_regression

from config import TOP_K_FEATURES, RANDOM_STATE


# ============================================================
# 0. Helper: build scalar target cho multi-output
# ============================================================

def build_selection_target_from_multi(Y_multi: np.ndarray, mode: str = "sum") -> np.ndarray:
    """
    Chuyển target multi output (N, H) thành scalar cho bước feature selection.

    mode:
      - "sum": tổng return 100 ngày
      - "mean": trung bình return 100 ngày
      - "endpoint": dùng return endpoint lp_{t+H} - lp_t (ở đây vẫn là sum)
    """
    if mode == "sum":
        return Y_multi.sum(axis=1)
    elif mode == "mean":
        return Y_multi.mean(axis=1)
    elif mode == "endpoint":
        # với log-return nhiều bước, endpoint = sum các bước
        return Y_multi.sum(axis=1)
    else:
        raise ValueError(f"Unknown mode for selection target: {mode}")


# ============================================================
# 1. Helper chung cho scoring
# ============================================================

def _safe_series(values, index) -> pd.Series:
    s = pd.Series(values, index=index, dtype=float)
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return s


def _normalize_score(s: pd.Series) -> pd.Series:
    """
    Chuẩn hóa score về [0,1]. Giả định s không âm (MI, |corr|, R2, stability).
    """
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    max_val = s.max()
    if max_val is None or max_val <= 0:
        return pd.Series(0.0, index=s.index)
    return s / max_val


def _safe_corr(x: np.ndarray, y: np.ndarray, min_samples: int = 30) -> float:
    """
    Corr Pearson tuyệt đối giữa 2 vector, xử lý NaN/inf, yêu cầu min_samples.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < min_samples:
        return 0.0

    x_m = x[mask] - x[mask].mean()
    y_m = y[mask] - y[mask].mean()
    denom = np.sqrt((x_m ** 2).sum()) * np.sqrt((y_m ** 2).sum())
    if denom <= 0:
        return 0.0
    return float((x_m * y_m).sum() / denom)


# ============================================================
# 2. Score 1: Mutual Information với target (smoothed)
# ============================================================

def _compute_mi_scores(
    df: pd.DataFrame,
    feature_cols: List[str],
    y: np.ndarray,
) -> pd.Series:
    """
    Mutual Information giữa mỗi feature và target liên tục y.

    - X: matrix (N, F) các feature (đã fillna)
    - y: vector (N,)
    """
    # y
    y = np.asarray(y, dtype=float)
    mask_y = np.isfinite(y)
    y_clean = y[mask_y]

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median(axis=0)
    X = X.fillna(medians)
    X = X.iloc[mask_y]

    if len(X) == 0 or len(X) != len(y_clean):
        return pd.Series(0.0, index=feature_cols, dtype=float)

    mi = mutual_info_regression(
        X.values,
        y_clean,
        random_state=RANDOM_STATE,
    )
    return _safe_series(mi, feature_cols)


# ============================================================
# 3. Score 2: Multi-scale correlation với target smoothed
# ============================================================

def _compute_multiscale_corr_scores(
    df: pd.DataFrame,
    feature_cols: List[str],
    y: np.ndarray,
) -> pd.Series:
    """
    Corr tuyệt đối giữa feature và nhiều phiên bản smoothed của y:
      - EMA 5, 20, 60
      - Rolling mean 40
      - Rolling median 50
    Score = trung bình |corr| trên các phiên bản.
    """
    y = np.asarray(y, dtype=float)
    y_series = pd.Series(y)

    y_ema5 = y_series.ewm(span=5, adjust=False).mean().values
    y_ema20 = y_series.ewm(span=20, adjust=False).mean().values
    y_ema60 = y_series.ewm(span=60, adjust=False).mean().values
    y_rm40 = y_series.rolling(window=40, min_periods=20).mean().values
    y_med50 = y_series.rolling(window=50, min_periods=25).median().values

    ys = [y_ema5, y_ema20, y_ema60, y_rm40, y_med50]

    scores = {}
    for feat in feature_cols:
        x = df[feat].values.astype(float)
        corrs = []
        for yy in ys:
            c = abs(_safe_corr(x, yy))
            corrs.append(c)
        scores[feat] = float(np.mean(corrs))

    return _safe_series(scores, feature_cols)


# ============================================================
# 4. Score 3: Stability Selection với ElasticNet trên các fold
# ============================================================

def _compute_stability_scores(
    df: pd.DataFrame,
    feature_cols: List[str],
    y: np.ndarray,
    folds: List[Dict],
    min_samples: int = 50,
) -> pd.Series:
    """
    Stability Selection đơn giản:
      - Với mỗi fold (train_mask):
          - Fit ElasticNet trên train
          - Đếm số lần coef != 0 cho mỗi feature
      - stability_score = counts / n_models
    """
    y = np.asarray(y, dtype=float)
    counts = {f: 0 for f in feature_cols}
    n_models = 0

    for fold in folds:
        train_mask = fold["train_mask"]
        X_train = df.loc[train_mask, feature_cols]
        y_train = y[train_mask]

        # drop dòng có NaN/inf
        X = X_train.replace([np.inf, -np.inf], np.nan)
        mask_rows = np.isfinite(y_train)
        mask_rows &= ~X.isna().any(axis=1).values

        if mask_rows.sum() < min_samples:
            continue

        X_clean = X.iloc[mask_rows].values
        y_clean = y_train[mask_rows]

        # chuẩn hóa feature về zero-mean unit-std cho ổn định hơn
        mean = X_clean.mean(axis=0)
        std = X_clean.std(axis=0)
        std[std == 0] = 1.0
        X_std = (X_clean - mean) / std

        enet = ElasticNet(
            alpha=1e-2,
            l1_ratio=0.7,
            random_state=RANDOM_STATE,
            max_iter=3000,
        )
        try:
            enet.fit(X_std, y_clean)
        except Exception:
            continue

        coef = enet.coef_
        for j, feat in enumerate(feature_cols):
            if abs(coef[j]) > 1e-6:
                counts[feat] += 1

        n_models += 1

    if n_models == 0:
        # không fit được model nào
        return pd.Series(0.0, index=feature_cols, dtype=float)

    stability = {f: counts[f] / float(n_models) for f in feature_cols}
    return _safe_series(stability, feature_cols)


# ============================================================
# 5. Score 4: Predictability Score (AR(1) R^2 của từng feature)
# ============================================================

def _compute_predictability_scores(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_samples: int = 50,
) -> pd.Series:
    """
    Predictability Score:
      - Với mỗi feature X:
          - Dùng AR(1): X_t ~ a * X_{t-1}
          - R^2 trên những điểm hợp lệ
      - Feature càng "có cấu trúc" (autocorrelation) thì R^2 càng cao.
    """
    scores = {}

    for feat in feature_cols:
        x = df[feat].values.astype(float)
        x_lag = np.roll(x, 1)
        mask = np.isfinite(x) & np.isfinite(x_lag)
        mask[0] = False

        if mask.sum() < min_samples:
            scores[feat] = 0.0
            continue

        X = x_lag[mask].reshape(-1, 1)
        y = x[mask]

        lr = LinearRegression()
        try:
            lr.fit(X, y)
            y_hat = lr.predict(X)
        except Exception:
            scores[feat] = 0.0
            continue

        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot <= 0:
            r2 = 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot

        scores[feat] = max(0.0, float(r2))

    return _safe_series(scores, feature_cols)


# ============================================================
# 6. Core: Advanced Feature Selection
# ============================================================

def _run_advanced_feature_ranking(
    df: pd.DataFrame,
    y: np.ndarray,
    feature_cols: List[str],
    folds: List[Dict],
    top_k: int,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Core logic: tính 4 score cho mỗi feature, rồi kết hợp.
      - MI với y
      - Multi-scale corr với y smoothed
      - Stability selection (ElasticNet, sử dụng folds)
      - Predictability AR(1) của bản thân feature
    """
    # đảm bảo feature_cols không rỗng
    if len(feature_cols) == 0:
        raise ValueError("feature_cols is empty in _run_advanced_feature_ranking")

    # Bỏ các cột hoàn toàn NaN
    valid_cols = []
    for f in feature_cols:
        col = df[f]
        if not col.replace([np.inf, -np.inf], np.nan).dropna().empty:
            valid_cols.append(f)
    feature_cols = valid_cols

    if len(feature_cols) == 0:
        raise ValueError("All feature columns are empty or NaN in _run_advanced_feature_ranking")

    # Score 1: Mutual Information
    mi_scores = _compute_mi_scores(df, feature_cols, y)

    # Score 2: Multi-scale corr
    corr_scores = _compute_multiscale_corr_scores(df, feature_cols, y)

    # Score 3: Stability
    stab_scores = _compute_stability_scores(df, feature_cols, y, folds)

    # Score 4: Predictability
    pred_scores = _compute_predictability_scores(df, feature_cols)

    # Chuẩn hóa từng score
    mi_norm = _normalize_score(mi_scores)
    corr_norm = _normalize_score(corr_scores)
    stab_norm = _normalize_score(stab_scores)
    pred_norm = _normalize_score(pred_scores)

    # Trộn score với trọng số
    w_mi = 0.3
    w_corr = 0.3
    w_stab = 0.3
    w_pred = 0.1

    final_score = (
        w_mi * mi_norm
        + w_corr * corr_norm
        + w_stab * stab_norm
        + w_pred * pred_norm
    )

    # build rank_df để debug/inspect
    rank_df = pd.DataFrame(
        {
            "mi_score": mi_scores,
            "mi_norm": mi_norm,
            "corr_score": corr_scores,
            "corr_norm": corr_norm,
            "stability_score": stab_scores,
            "stability_norm": stab_norm,
            "predictability_score": pred_scores,
            "predictability_norm": pred_norm,
            "final_score": final_score,
        }
    )

    rank_df = rank_df.sort_values("final_score", ascending=False)
    selected_features = rank_df.index.tolist()[:top_k]

    return selected_features, rank_df


# ============================================================
# 7. Public API: Pipeline 1 Direct 100d (scalar)
# ============================================================

def run_feature_selection_direct(
    df_direct: pd.DataFrame,
    folds: List[Dict],
    feature_cols: List[str],
    top_k: int = TOP_K_FEATURES,
    base_model_for_importance: ElasticNet = None,  # giữ tham số để không vỡ API cũ
    min_folds_used: int = 1,                       # không dùng nữa nhưng giữ signature
) -> Tuple[List[str], pd.DataFrame]:
    """
    Advanced Feature Selection cho Pipeline 1 Direct 100d.

    Chiến lược:
      - Không dùng model importance trên y_direct raw
      - Thay vào đó:
          + Mutual Information với y_direct
          + Corr với y_direct đã smooth multi-scale
          + Stability selection (ElasticNet, nhiều fold)
          + Predictability (AR(1) R^2 của từng feature)
      - Kết hợp 4 score thành final_score và chọn top_k.

    df_direct:
      - DataFrame đã build features, có cột:
          lp, y_direct, time, feature_cols

    folds:
      - list dict, mỗi dict có "train_mask" và "val_mask" (bool index)

    feature_cols:
      - danh sách tên feature để ranking
    """
    if "y_direct" not in df_direct.columns:
        raise ValueError("df_direct must contain 'y_direct' column for run_feature_selection_direct")

    y = df_direct["y_direct"].values.astype(float)

    selected_features, rank_df = _run_advanced_feature_ranking(
        df=df_direct,
        y=y,
        feature_cols=feature_cols,
        folds=folds,
        top_k=top_k,
    )

    return selected_features, rank_df


# ============================================================
# 8. Public API: Pipeline 2 Multi-step
# ============================================================

def run_feature_selection_multi(
    df_multi: pd.DataFrame,
    Y_multi: np.ndarray,
    folds: List[Dict],
    feature_cols: List[str],
    top_k: int = TOP_K_FEATURES,
    base_model_for_importance: ElasticNet = None,  # giữ để không vỡ API
    min_folds_used: int = 1,
    selection_target_mode: str = "sum",
) -> Tuple[List[str], pd.DataFrame]:
    """
    Advanced Feature Selection cho Pipeline 2 Multi-step.

    df_multi:
      - DataFrame đã build features, align index với Y_multi
    Y_multi:
      - numpy array shape (N, H)
    folds:
      - output của make_folds trên df_multi["time"]
    feature_cols:
      - list tên feature ban đầu
    """
    y_scalar_all = build_selection_target_from_multi(Y_multi, mode=selection_target_mode)

    selected_features, rank_df = _run_advanced_feature_ranking(
        df=df_multi,
        y=y_scalar_all,
        feature_cols=feature_cols,
        folds=folds,
        top_k=top_k,
    )

    return selected_features, rank_df
