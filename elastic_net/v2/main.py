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

    # =========================================
    # Calibration 1-step trên toàn bộ validation
    # =========================================
    cal = LinearRegression()
    cal.fit(y_pred_val.reshape(-1, 1), y_true_val)
    y_pred_val_cal = cal.predict(y_pred_val.reshape(-1, 1)).reshape(-1)

    mse_before = mean_squared_error(y_true_val, y_pred_val)
    mse_after = mean_squared_error(y_true_val, y_pred_val_cal)

    print("\nCalibration 1-step trên tập validation:")
    print(f"  a = {cal.intercept_:.6e}, b = {cal.coef_[0]:.6f}")
    print(f"  MSE trước calibration: {mse_before:.8f}")
    print(f"  MSE sau   calibration: {mse_after:.8f}")
    
        # =========================================
    # Tính blend weight trên 1-step PRICE
    # =========================================
    print("\nTối ưu ensemble weight trên PRICE 1-step:")

    # Các index trong validation dùng cho đánh giá
    val_idx_all = np.where(mask_val_used)[0]

    price_true_list = []
    price_naive_list = []
    price_model_list = []

    for idx_feat, y_pred_raw, y_pred_cal in zip(
        val_idx_all, y_pred_val, y_pred_val_cal
    ):
        t_time = feat_df.loc[idx_feat, "time"]
        df_idx_arr = df.index[df["time"] == t_time]
        if len(df_idx_arr) == 0:
            continue
        df_idx = int(df_idx_arr[0])

        # Cần có giá ngày t và t+1
        if df_idx + 1 >= len(df):
            continue

        price_t = float(df["close"].iloc[df_idx])
        price_next_true = float(df["close"].iloc[df_idx + 1])

        # Naive: giữ nguyên giá t
        price_next_naive = price_t

        # Model: dùng return đã calibrated một bước
        price_next_model = price_t * float(np.exp(y_pred_cal))

        price_true_list.append(price_next_true)
        price_naive_list.append(price_next_naive)
        price_model_list.append(price_next_model)

    price_true_arr = np.array(price_true_list, dtype=float)
    price_naive_arr = np.array(price_naive_list, dtype=float)
    price_model_arr = np.array(price_model_list, dtype=float)

    if len(price_true_arr) == 0:
        raise RuntimeError("Không có đủ điểm để tối ưu ensemble 1-step")

    # Grid search đơn giản w trong [0, 1]
    best_w = 0.0
    best_mse = np.inf
    for w in np.linspace(0.0, 1.0, 21):
        price_blend = w * price_naive_arr + (1.0 - w) * price_model_arr
        mse_w = mean_squared_error(price_true_arr, price_blend)
        if mse_w < best_mse:
            best_mse = mse_w
            best_w = w

    print(f"  Naive PRICE MSE: {mean_squared_error(price_true_arr, price_naive_arr):.6f}")
    print(f"  Model PRICE MSE: {mean_squared_error(price_true_arr, price_model_arr):.6f}")
    print(f"  Best w_naive = {best_w:.2f}, w_model = {1.0 - best_w:.2f}, MSE = {best_mse:.6f}")


    # Bước 11: fit ElasticNet cuối cùng trên toàn bộ dữ liệu
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

    # Áp calibration 1-step cho chuỗi return tương lai
    # (dùng cal đã fit trên toàn bộ validation)
    future_returns_cal = cal.predict(future_returns_raw.reshape(-1, 1)).reshape(-1)

    # Chuyển thành path giá theo model
    last_price = df["close"].iloc[-1]
    price_future_model = returns_to_prices(
        last_price=last_price,
        future_returns=future_returns_cal,
    )

    # Path giá naive: giữ nguyên giá cuối cùng
    price_future_naive = np.full_like(price_future_model, last_price, dtype=float)

    # Ensemble với trọng số best_w học được từ PRICE 1-step
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
