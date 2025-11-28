# main_pipeline1_direct.py

import numpy as np
import pandas as pd

from targets_direct import build_direct_100d_target
from models_direct import MODEL_REGISTRY
from tuning_direct import (
    tune_model_direct,
    collect_validation_predictions_direct,
)
from ensemble import (
    tune_ensemble_weights_random_search,
    tune_ensemble_shrinkage,
)

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import (
    load_data,
    add_base_series,
    ensure_business_day_indexing,
    winsorize_series,
)

from config import HORIZON, TRAIN_CSV
from submission import make_submission
from features import build_features
from splits import make_folds, get_test_indices
from feature_selection import run_feature_selection_direct


def run_pipeline1_direct(train_csv: str, submission_output: str) -> None:
    # 0. Load và base series
    df = load_data(train_csv)
    df = ensure_business_day_indexing(df)
    df = add_base_series(df)

    # 1. Winsorize ret_1d, vol_log (clip theo toàn period hoặc mask train)
    train_mask_all = pd.Series(True, index=df.index)
    df = winsorize_series(df, train_mask_all, cols=["ret_1d", "vol_log"])

    # 2. Feature engineering
    df_feat = build_features(df)

    # 3. Build target direct 100d
    df_target, y_direct = build_direct_100d_target(df_feat, horizon=HORIZON)
    df_target["y_direct"] = y_direct

    # 4. Define feature_cols ban đầu
    drop_cols = ["time", "close", "lp", "y_direct"]
    feature_cols_all = [c for c in df_target.columns if c not in drop_cols]

    # 5. Time series CV folds
    folds = make_folds(df_target["time"], n_folds=3, horizon=HORIZON)

    # 6. Feature selection cho direct 100d (pipeline 1)
    selected_features, rank_df = run_feature_selection_direct(
        df_direct=df_target,
        folds=folds,
        feature_cols=feature_cols_all,
        top_k=80,
        min_folds_used=1,
    )
    feature_cols = selected_features
    print("Selected features (direct 100d):", feature_cols)

    # 7. Chạy từng model nhỏ, hyperparameter tuning từng model
    #    Model list tùy chọn, bạn có thể bỏ bớt để chạy được trên Kaggle
    model_list = [
        "elasticnet",
        "ridge",
        "xgboost",
        "lgbm",
        "random_forest",
        "gbdt",
        # "rolling_elasticnet",
        # "kalman",
        # "dlinear",
        # "nlinear",
    ]

    best_configs = {}
    best_scores = {}

    for model_name in model_list:
        # Skip những model chưa implement nếu cần
        if model_name not in MODEL_REGISTRY:
            print(f"Model {model_name} not found in registry, skip")
            continue

        try:
            cfg, score = tune_model_direct(
                model_name=model_name,
                df_direct=df_target,
                folds=folds,
                feature_cols=feature_cols,
                horizon=HORIZON,
                max_configs=None,  # hoặc giới hạn để đỡ nặng
            )
            best_configs[model_name] = cfg
            best_scores[model_name] = score
        except NotImplementedError as e:
            print(f"Model {model_name} not implemented fully, skip:", e)
        except ImportError as e:
            print(f"Model {model_name} missing dependency, skip:", e)

    print("Best configs per model:", best_configs)
    print("CV scores per model:", best_scores)

    # 8. Chạy lại CV với config tốt nhất để collect validation predictions
    #    Chuẩn bị data cho ensemble
    price_true_all = None
    price_hat_matrix_list = []
    used_models = []

    for model_name, cfg in best_configs.items():
        print(f"Collect validation predictions for model {model_name}...")
        pt, ph = collect_validation_predictions_direct(
            model_name=model_name,
            best_config=cfg,
            df_direct=df_target,
            folds=folds,
            feature_cols=feature_cols,
            horizon=HORIZON,
        )

        if price_true_all is None:
            price_true_all = pt
        else:
            # sanity check
            if not np.allclose(price_true_all, pt):
                print("Warning: price_true_all mismatch across models; continue anyway")

        price_hat_matrix_list.append(ph)
        used_models.append(model_name)

    if price_true_all is None or len(price_hat_matrix_list) == 0:
        raise RuntimeError("No model predictions collected for ensemble")

    price_hat_matrix = np.vstack(price_hat_matrix_list).T  # shape (N, M)
    print("Ensemble uses models:", used_models)

    # 9. Ensemble - tuning lần 1: tìm weight tối ưu trên simplex
    w_star = tune_ensemble_weights_random_search(
        price_hat_matrix=price_hat_matrix,
        price_true=price_true_all,
        n_samples=2000,
        l2_shrink=0.0,
    )
    print("w_star (before shrinkage):", dict(zip(used_models, w_star)))

    # 10. Ensemble - tuning lần 2: shrink weight về equal weight
    best_lambda, w_opt = tune_ensemble_shrinkage(
        price_hat_matrix=price_hat_matrix,
        price_true=price_true_all,
        w_star=w_star,
        shrink_values=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    print("best_lambda:", best_lambda)
    print("w_opt (final ensemble weights):", dict(zip(used_models, w_opt)))

    # 11. Train full từng model trên train_val (trừ 100 ngày test cuối)
    #     và dự báo R_hat_test, rồi ensemble theo w_opt trên test
    test_info = get_test_indices(df_target["time"], horizon=HORIZON)
    train_val_mask = test_info["train_val_mask"]
    last_index_before_test = np.where(train_val_mask)[0][-1]

    df_train_val = df_target.loc[train_val_mask]
    X_train_val = df_train_val[feature_cols]
    y_train_val = df_train_val["y_direct"].values

    lp_full = df_target["lp"].values
    R_hat_test_all = []

    from models_direct import MODEL_REGISTRY as REG

    for i, model_name in enumerate(used_models):
        cfg = best_configs[model_name]
        ModelClass = REG[model_name]
        model = ModelClass(cfg)

        # Fit full
        model.fit(X_train_val, y_train_val)

        # Predict từ điểm cuối trước test
        X_last = df_target.loc[[last_index_before_test], feature_cols]
        R_hat_test = model.predict_100day_return(X_last)[0]
        R_hat_test_all.append(R_hat_test)

    R_hat_test_all = np.array(R_hat_test_all)  # shape (M,)
    R_hat_test_ensemble = float(np.dot(w_opt, R_hat_test_all))

    # 12. Convert R_hat_test_ensemble thành path 100 ngày (chia đều)
    lp_last = df_target.loc[last_index_before_test, "lp"]
    r_daily = R_hat_test_ensemble / HORIZON
    lp_path = lp_last + np.arange(1, HORIZON + 1) * r_daily
    price_path = np.exp(lp_path)

    # 13. Xuất submission
    make_submission(price_path, submission_output)
    print("Submission saved to:", submission_output)
    
if __name__ == "__main__":
    run_pipeline1_direct(
        train_csv=TRAIN_CSV,
        submission_output="submission_pipeline1_direct.csv",
    )