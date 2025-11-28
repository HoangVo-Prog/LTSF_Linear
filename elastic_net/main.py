import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from config import CONFIG, FEATURE_NAMES
from data_utils import set_seed, find_data_path, add_base_returns, winsorize_returns
from features import build_features
from models import (
    grid_search_window_and_reg,
    grid_search_alpha_l1,
    rolling_elasticnet_forecast,
    fit_final_elasticnet,
    evaluate_predictions,
)
from forecasting import forecast_future_returns, returns_to_prices


# =========================================
# Pipeline chính
# =========================================

def main() -> None:
    # Bước 0: đặt seed
    set_seed(CONFIG["seed"])

    # Bước 1: đọc dữ liệu thô và sort theo thời gian
    data_path = find_data_path("FPT_train.csv")
    df = pd.read_csv(data_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    print("Raw data:", df.shape, df["time"].min(), "->", df["time"].max())

    # Bước 2: tính return cơ bản và winsorize trên phần train
    df = add_base_returns(df)
    df = winsorize_returns(
        df,
        val_start=CONFIG["val_start"],
        lower_q=CONFIG["clip_lower_q"],
        upper_q=CONFIG["clip_upper_q"],
    )

    # Bước 3: build features và target y
    feat_df = build_features(df)
    val_start_ts = pd.Timestamp(CONFIG["val_start"])
    feat_df["is_val"] = feat_df["time"] >= val_start_ts

    feature_cols = FEATURE_NAMES
    X_all = feat_df[feature_cols].values
    y_all = feat_df["y"].values
    val_mask = feat_df["is_val"].values
    train_mask = ~val_mask

    # Bước 4: scale feature theo thống kê của train
    scaler = StandardScaler()
    scaler.fit(X_all[train_mask])
    X_all_scaled = scaler.transform(X_all)

    print("Number of samples used to fit scaler:", train_mask.sum())

    # Bước 5: grid search lần 1 (tìm window, alpha, l1_ratio thô)
    print("\nGrid search 1 - chọn window_size, window_type, alpha, l1_ratio cơ bản:")
    results_1, best_1 = grid_search_window_and_reg(
        X_all_scaled=X_all_scaled,
        y_all=y_all,
        val_mask=val_mask,
        seed=CONFIG["seed"],
    )
    print("Best config (coarse search):", best_1)
    print(results_1.head(10))

    # Bước 6: grid search lần 2 (tinh chỉnh alpha, l1_ratio với window cố định)
    print("\nGrid search 2 - tinh chỉnh alpha và l1_ratio:")
    results_2, best_2 = grid_search_alpha_l1(
        X_all_scaled=X_all_scaled,
        y_all=y_all,
        val_mask=val_mask,
        base_window_size=best_1["window_size"],
        base_window_type=best_1["window_type"],
        seed=CONFIG["seed"],
    )
    print("Best config (fine search):", best_2)
    print(results_2.head(10))

        # Bước 7: dùng cấu hình tốt nhất để rolling predict 1-step trên toàn bộ chuỗi (chỉ để log)
    preds_all = rolling_elasticnet_forecast(
        X=X_all_scaled,
        y=y_all,
        window_size=best_2["window_size"],
        alpha=best_2["alpha"],
        l1_ratio=best_2["l1_ratio"],
        window_type=best_2["window_type"],
        random_state=CONFIG["seed"],
    )
    mask_val_used = val_mask & ~np.isnan(preds_all)

    y_true_val = y_all[mask_val_used]
    y_pred_val = preds_all[mask_val_used]

    print("\nKết quả 1-step-ahead return trên tập validation (log cho tham khảo):")
    m0 = evaluate_predictions(y_true_val, y_pred_val)
    print(f"MSE: {m0['mse']:.8f}, MAE: {m0['mae']:.8f}, n_val_used: {mask_val_used.sum()}")

    # ============================================================
    # Bước 8: multi step validation mô phỏng đúng pipeline submit
    # ============================================================
    print("\nMulti step validation mô phỏng pipeline submit:")

    val_horizon = int(CONFIG.get("val_horizon", 50))
    val_num_anchors = int(CONFIG.get("val_num_anchors", 10))

    times_all = feat_df["time"].values
    val_idx = np.where(feat_df["is_val"].values)[0]

    # Chỉ chọn anchor sao cho còn đủ val_horizon ngày phía sau
    max_anchor_idx = len(feat_df) - val_horizon - 1
    val_idx = val_idx[val_idx <= max_anchor_idx]

    if len(val_idx) == 0:
        raise RuntimeError("Không đủ dữ liệu validation để chạy multi step validation")

    # Lấy tối đa val_num_anchors anchor, trải đều trong đoạn validation
    if len(val_idx) > val_num_anchors:
        anchor_indices = np.linspace(0, len(val_idx) - 1, val_num_anchors).round().astype(int)
        anchor_idx_list = val_idx[anchor_indices]
    else:
        anchor_idx_list = val_idx

    print(f"Số anchor được dùng cho multi step validation: {len(anchor_idx_list)}")

    all_pred_rets = []
    all_true_rets = []
    all_true_prices = []
    all_model_prices_uncal = []
    all_naive_prices = []
    anchor_last_prices = []

    for anchor_idx in anchor_idx_list:
        anchor_time = feat_df.loc[anchor_idx, "time"]
        print(f"  Anchor tại ngày: {anchor_time}")

        # Dữ liệu feature và target dùng để train model tại anchor
        X_hist_scaled = X_all_scaled[: anchor_idx + 1]
        y_hist = y_all[: anchor_idx + 1]

        # Train static ElasticNet theo đúng fit_final_elasticnet
        model_anchor = fit_final_elasticnet(
            X_scaled=X_hist_scaled,
            y=y_hist,
            window_size=best_2["window_size"],
            alpha=best_2["alpha"],
            l1_ratio=best_2["l1_ratio"],
            window_type=best_2["window_type"],
            random_state=CONFIG["seed"],
        )

        # Map anchor_time sang index trong df gốc
        df_anchor_idx = df.index[df["time"] == anchor_time]
        if len(df_anchor_idx) == 0:
            # Nếu không map được thẳng (do dropna trong features) thì bỏ anchor này
            print("    Không tìm thấy anchor trong df gốc, bỏ qua.")
            continue
        df_anchor_idx = int(df_anchor_idx[0])

        df_hist = df.iloc[: df_anchor_idx + 1]

        # Dự đoán multi step returns từ anchor
        pred_rets = forecast_future_returns(
            model=model_anchor,
            scaler=scaler,
            df=df_hist,
            steps=val_horizon,
        )

        # True future returns (raw ret_1d) và giá
        df_future = df.iloc[df_anchor_idx + 1 : df_anchor_idx + 1 + val_horizon]
        true_rets = df_future["ret_1d"].values.astype(float)
        true_prices = df_future["close"].values.astype(float)

        if len(true_rets) < val_horizon:
            # Không đủ dữ liệu tương lai, bỏ anchor
            print("    Không đủ dữ liệu tương lai cho horizon, bỏ qua anchor.")
            continue

        last_price = df_hist["close"].iloc[-1]
        anchor_last_prices.append(last_price)

        # Xây dựng price path từ returns
        price_true_path = returns_to_prices(last_price=last_price, future_returns=true_rets)
        price_model_uncal_path = returns_to_prices(last_price=last_price, future_returns=pred_rets)

        price_naive_path = np.full_like(price_true_path, last_price, dtype=float)

        all_pred_rets.append(pred_rets)
        all_true_rets.append(true_rets)
        all_true_prices.append(price_true_path)
        all_model_prices_uncal.append(price_model_uncal_path)
        all_naive_prices.append(price_naive_path)

    if len(all_pred_rets) == 0:
        raise RuntimeError("Không có anchor hợp lệ để multi step validation")

    # Gộp các anchor lại
    y_true_flat = np.concatenate(all_true_rets)
    y_pred_flat = np.concatenate(all_pred_rets)

    # =========================================
    # Bước 9: calibration trên multi step returns
    # =========================================
    print("\nCalibrating return predictions trên multi step validation:")

    cal = LinearRegression()
    cal.fit(y_pred_flat.reshape(-1, 1), y_true_flat)
    y_pred_cal_flat = cal.predict(y_pred_flat.reshape(-1, 1))

    mse_before_cal = mean_squared_error(y_true_flat, y_pred_flat)
    mse_after_cal = mean_squared_error(y_true_flat, y_pred_cal_flat)

    print(f"Calibration coefficients (multi step): a={cal.intercept_:.6e}, b={cal.coef_[0]:.6f}")
    print(f"Return MSE trước calibration: {mse_before_cal:.8f}")
    print(f"Return MSE sau   calibration: {mse_after_cal:.8f}")

    # Áp calibration lên từng anchor để tính price path
    all_model_prices_cal = []
    for preds_anchor, last_price in zip(all_pred_rets, anchor_last_prices):
        preds_cal = cal.predict(preds_anchor.reshape(-1, 1)).reshape(-1)
        price_model_cal_path = returns_to_prices(last_price=last_price, future_returns=preds_cal)
        all_model_prices_cal.append(price_model_cal_path)

    # Gộp price path để đánh giá
    prices_true_flat = np.concatenate(all_true_prices)
    prices_model_cal_flat = np.concatenate(all_model_prices_cal)
    prices_naive_flat = np.concatenate(all_naive_prices)

    mse_price_naive = mean_squared_error(prices_true_flat, prices_naive_flat)
    mse_price_model = mean_squared_error(prices_true_flat, prices_model_cal_flat)

    print("\nValidation multi step PRICE MSE:")
    print(f"  Naive:  {mse_price_naive:.6f}")
    print(f"  Model (calibrated):  {mse_price_model:.6f}")

    # =========================================
    # Bước 10: ensemble trên multi step price path
    # =========================================
    best_w = 0.0
    best_price_mse = np.inf
    for w in np.linspace(0.0, 1.0, 21):
        p_blend = w * prices_naive_flat + (1.0 - w) * prices_model_cal_flat
        mse_blend = mean_squared_error(prices_true_flat, p_blend)
        if mse_blend < best_price_mse:
            best_price_mse = mse_blend
            best_w = w

    print("\nBest ensemble weight trên multi step price validation:")
    print(f"  w_naive = {best_w:.2f}, w_model = {1.0 - best_w:.2f}, MSE = {best_price_mse:.6f}")

        # Bước 11: fit model cuối cùng trên toàn bộ dữ liệu
    print("\nFit ElasticNet cuối cùng trên toàn bộ dữ liệu:")
    final_model = fit_final_elasticnet(
        X_scaled=X_all_scaled,
        y=y_all,
        window_size=best_2["window_size"],
        alpha=best_2["alpha"],
        l1_ratio=best_2["l1_ratio"],
        window_type=best_2["window_type"],
        random_state=CONFIG["seed"],
    )

    # Bước 12: dự đoán tương lai nhiều bước theo kiểu autoregressive
    print("Forecasting future returns...")
    future_returns_raw = forecast_future_returns(
        model=final_model,
        scaler=scaler,
        df=df,
        steps=CONFIG["forecast_steps"],
    )

    # Áp calibration cho chuỗi return tương lai (dùng cal đã fit trên multi step)
    future_returns_cal = cal.predict(future_returns_raw.reshape(-1, 1)).reshape(-1)

    # Chuyển thành path giá theo model
    last_price = df["close"].iloc[-1]
    price_future_model = returns_to_prices(
        last_price=last_price,
        future_returns=future_returns_cal,
    )

    # Path giá naive: luôn giữ nguyên giá cuối cùng
    price_future_naive = np.full_like(price_future_model, last_price, dtype=float)

    # Ensemble với trọng số đã tối ưu trên multi step validation
    price_future_final = best_w * price_future_naive + (1.0 - best_w) * price_future_model

    future_signals_df = pd.DataFrame(
        {
            "step": np.arange(1, CONFIG["forecast_steps"] + 1),
            "ret_raw": future_returns_raw,
            "ret_cal": future_returns_cal,
            "price_model": price_future_model,
            "price_naive": price_future_naive,
            "price_final": price_future_final,
        }
    )
    print("\nPreview 100 ngày tương lai (price_final):")
    print(future_signals_df.head(10))

    # Bước 13: tạo file submission
    submission = pd.DataFrame(
        {
            "id": np.arange(1, CONFIG["forecast_steps"] + 1),
            "close": price_future_final.astype(float),
        }
    )
    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)
    print("\nSaved submission file to:", submission_path)
    print(submission.head())


if __name__ == "__main__":
    main()
