import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor

from config import HORIZON
from data_utils import load_data, add_base_series, ensure_business_day_indexing, winsorize_series
from features import build_features
from splits import make_folds, get_test_indices
from targets_multi import build_multi_step_target
from models_multi import MultiStepModel
from evaluation_multi import build_price_path_from_returns, mse_full_path
from submission import make_submission


def run_pipeline2_multi(train_csv: str, submission_output: str) -> None:
    df = load_data(train_csv)
    df = ensure_business_day_indexing(df)
    df = add_base_series(df)

    train_mask_all = pd.Series(True, index=df.index)
    df = winsorize_series(df, train_mask_all, cols=["ret_1d", "vol_log"])

    # Build features
    df_feat = build_features(df)

    # Build multi step target
    df_target, Y_multi = build_multi_step_target(df_feat, horizon=HORIZON)
    # Lưu để align trong evaluation Multi
    df_target = df_target.reset_index(drop=True)
    Y_multi = Y_multi  # shape (N, H)

    feature_cols = [c for c in df_target.columns if c not in ["time", "close", "lp"]]

    folds = make_folds(df_target["time"], n_folds=3, horizon=HORIZON)

    base_model = MultiOutputRegressor(ElasticNet())
    model = MultiStepModel(base_model)

    # CV
    scores = []
    for fold in folds:
        train_mask = fold["train_mask"]
        val_eval_start_indices = fold["val_eval_start_indices"]

        df_train = df_target.loc[train_mask]
        X_train = df_train[feature_cols]
        Y_train = Y_multi[df_train.index]

        model.fit(X_train, Y_train)

        X_val_eval = df_target.loc[val_eval_start_indices, feature_cols]
        r_hat_all = model.predict_path(X_val_eval)

        price_dict = build_price_path_from_returns(
            df_target, val_eval_start_indices, r_hat_all, horizon=HORIZON
        )
        score = mse_full_path(price_dict["price_true_paths"], price_dict["price_hat_paths"])
        scores.append(score)

    print("CV scores:", scores, "mean:", np.mean(scores))

    # Train full trên train_val (trừ 100 ngày cuối)
    test_info = get_test_indices(df_target["time"], horizon=HORIZON)
    train_val_mask = test_info["train_val_mask"]
    last_index_before_test = np.where(train_val_mask)[0][-1]

    df_train_val = df_target.loc[train_val_mask]
    X_train_val = df_train_val[feature_cols]
    Y_train_val = Y_multi[df_train_val.index]

    model.fit(X_train_val, Y_train_val)

    # Dự báo path cho 100 ngày tương lai
    X_last = df_target.loc[[last_index_before_test], feature_cols]
    r_hat_vec = model.predict_path(X_last)[0]

    lp_last = df_target.loc[last_index_before_test, "lp"]
    lp_path = []
    curr_lp = lp_last
    for k in range(HORIZON):
        curr_lp = curr_lp + r_hat_vec[k]
        lp_path.append(curr_lp)
    price_path = np.exp(np.array(lp_path))

    make_submission(price_path, submission_output)
