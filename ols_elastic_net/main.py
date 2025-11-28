# main.py

import numpy as np
import pandas as pd

from config import (
    TRAIN_CSV,
    SUBMISSION_TEMPLATE_CSV,
    OUTPUT_SUBMISSION_CSV,
    TRAIN_END_DATE,
    TREND_POLY_DEGREE,
    RESIDUAL_MODEL_TYPE,
    TOP_K_FEATURES,
    N_RANDOM_SEARCH,
    FORECAST_STEPS,
)

from data_utils import load_price_data, train_val_split
from trend_model import TrendModel
from features import add_technical_features, make_supervised_residual_dataset
from residual_model import ResidualModel
from feature_selection import compute_feature_importance, select_top_k_features, filter_feature_matrix
from optimize_residual import random_search_elasticnet, random_search_ridge
from forecast import forecast_future_prices

pd.set_option("compute.use_numexpr", False)



def evaluate_path_mse_on_validation(
    df_full: pd.DataFrame,
    trend_model: TrendModel,
    residual_model: ResidualModel,
    feature_names: list,
    start_date: str,
    horizon: int = 100,
) -> None:
    """
    Evaluate 100-step recursive forecast on a validation segment.

    start_date:
      We use all data up to and including this date as "history".
      Then we forecast `horizon` days and compare with the actual
      next `horizon` closes in df_full.
    """
    start_ts = pd.Timestamp(start_date)
    df_hist = df_full[df_full["time"] <= start_ts].copy()
    df_future_true = df_full[df_full["time"] > start_ts].copy()

    df_future_true = df_future_true.sort_values("time").reset_index(drop=True)

    if len(df_future_true) < horizon:
        print(f"[WARN] Not enough future data after {start_date} for horizon={horizon}")
        return

    preds = forecast_future_prices(
        df_hist_raw=df_hist,
        trend_model=trend_model,
        residual_model=residual_model,
        feature_names=feature_names,
        steps=horizon,
    )

    true_prices = df_future_true["close"].iloc[:horizon].values.astype(float)

    mse_path = np.mean((preds - true_prices) ** 2)
    print(f"Path MSE on validation (start={start_date}, horizon={horizon}): {mse_path:.4f}")
    print(f"True price range:   min={true_prices.min():.2f}, max={true_prices.max():.2f}")
    print(f"Predicted range:    min={preds.min():.2f},      max={preds.max():.2f}")
    

def main():
    # 1. Load data
    df = load_price_data(TRAIN_CSV)
    train_df, val_df = train_val_split(df, TRAIN_END_DATE)

    print(f"Train period: {train_df['time'].min()} -> {train_df['time'].max()}, n={len(train_df)}")
    print(f"Val period:   {val_df['time'].min()} -> {val_df['time'].max()}, n={len(val_df)}")

    # 2. Fit trend model on full history (2020–2025)
    trend_model = TrendModel(degree=TREND_POLY_DEGREE)
    trend_model.fit(df)

    # 3. Add trend and residual, then technical features for full data
    df_trend_all = trend_model.add_trend_and_residual(df)
    df_feat_all = add_technical_features(df_trend_all)

        
    # 4. Build supervised residual dataset
    X_all, y_all, feature_names_raw = make_supervised_residual_dataset(df_feat_all)

    # align X, y with train and val by date
    supervised_df = df_feat_all.loc[X_all.index, ["time"]].copy()
    supervised_df["target"] = y_all
    supervised_df = supervised_df.join(X_all)

    train_mask = supervised_df["time"] < pd.Timestamp(TRAIN_END_DATE)
    val_mask = supervised_df["time"] >= pd.Timestamp(TRAIN_END_DATE)

    # Use the actual columns from X_all to avoid any mismatch
    used_feature_names = list(X_all.columns)

    X_train = supervised_df.loc[train_mask, used_feature_names]
    y_train = supervised_df.loc[train_mask, "target"]
    X_val = supervised_df.loc[val_mask, used_feature_names]
    y_val = supervised_df.loc[val_mask, "target"]

    print(f"Supervised train shape: {X_train.shape}, val shape: {X_val.shape}")


    # 5. Initial residual model fit with default hyperparameters
    if RESIDUAL_MODEL_TYPE == "elasticnet":
        base_model = ResidualModel(
            model_type="elasticnet",
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=10000,
        )
    else:
        base_model = ResidualModel(
            model_type="ridge",
            alpha=1.0,
        )

    base_model.fit(X_train, y_train)
    mse_train = base_model.score_mse(X_train, y_train)
    mse_val = base_model.score_mse(X_val, y_val)
    print(f"Base residual model MSE train={mse_train:.6f}, val={mse_val:.6f}")

    # 6. Feature importance on validation
    importances = compute_feature_importance(base_model, X_val, y_val, n_repeats=10)

    # Use the actual columns of X_val so lengths must match permutation_importance
    feature_names_val = list(X_val.columns)

    print(f"Permutation importance: n_features={len(feature_names_val)}, "
        f"importances_len={len(importances)}")

    top_features = select_top_k_features(importances, feature_names_val, TOP_K_FEATURES)

    print("Top features:")
    for f in top_features:
        print("  ", f)


    # 7. Restrict to core features
    X_train_core = filter_feature_matrix(X_train, top_features)
    X_val_core = filter_feature_matrix(X_val, top_features)

    # 8. Hyperparameter random search on core features
    if RESIDUAL_MODEL_TYPE == "elasticnet":
        best_model, best_params = random_search_elasticnet(
            X_train_core,
            y_train,
            X_val_core,
            y_val,
            n_trials=N_RANDOM_SEARCH,
        )
    else:
        best_model, best_params = random_search_ridge(
            X_train_core,
            y_train,
            X_val_core,
            y_val,
            n_trials=N_RANDOM_SEARCH,
        )

    print("Best residual hyperparameters:", best_params)
    print(
        "Best residual MSE train={:.6f}, val={:.6f}".format(
            best_model.score_mse(X_train_core, y_train),
            best_model.score_mse(X_val_core, y_val),
        )
    )

    # 9. Refit final residual model on all supervised data (train + val) using core features
    X_all_core = filter_feature_matrix(X_all, top_features)
    y_all_core = y_all.loc[X_all_core.index]

    final_residual_model = ResidualModel(
        model_type=best_model.model_type,
        **{k: getattr(best_model.model, k) for k in best_params.keys()},
    )
    final_residual_model.fit(X_all_core, y_all_core)

    print("Final residual model fit on full supervised dataset.")
    
    # 10. Optional: sanity check 100-step path on validation
    evaluate_path_mse_on_validation(
        df_full=df,
        trend_model=trend_model,
        residual_model=final_residual_model,
        feature_names=top_features,
        start_date="2024-01-02",  # choose any date in validation with >= 100 future days
        horizon=100,
    )

    # 11. Forecast future prices (100 steps) for submission
    preds_future = forecast_future_prices(
        df_hist_raw=df,
        trend_model=trend_model,
        residual_model=final_residual_model,
        feature_names=top_features,
        steps=FORECAST_STEPS,
    )

    print(
        "Future price path for submission: min={:.2f}, max={:.2f}".format(
            preds_future.min(), preds_future.max()
        )
    )


    # 10. Forecast future prices (100 steps)
    preds_future = forecast_future_prices(
        df_hist_raw=df,
        trend_model=trend_model,
        residual_model=final_residual_model,
        feature_names=top_features,
        steps=FORECAST_STEPS,
    )

    # 11. Build submission
    sub_template = pd.read_csv(SUBMISSION_TEMPLATE_CSV)
    if len(sub_template) != FORECAST_STEPS:
        raise ValueError(
            f"Template has {len(sub_template)} rows but FORECAST_STEPS={FORECAST_STEPS}"
        )

    submission = sub_template.copy()
    submission["close"] = preds_future.astype(float)
    submission.to_csv(OUTPUT_SUBMISSION_CSV, index=False)
    print(f"Saved submission to {OUTPUT_SUBMISSION_CSV}")


if __name__ == "__main__":
    main()
