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
    price_all = feat_df["close"].values
    val_mask = feat_df["is_val"].values
    train_mask = ~val_mask

    # Bước 4: scale feature theo thống kê của train
    scaler = StandardScaler()
    scaler.fit(X_all[train_mask])
    X_all_scaled = scaler.transform(X_all)

    print("Number of samples used to fit scaler:", train_mask.sum())

    # Bước 5: grid search lần 1 (tìm window, alpha, l1_ratio thô) với MSE PRICE
    print("\nGrid search 1 - chọn window_size, window_type, alpha, l1_ratio cơ bản (score: price MSE 1-step):")
    results_1, best_1 = grid_search_window_and_reg(
        X_all_scaled=X_all_scaled,
        y_all=y_all,
        val_mask=val_mask,
        price_all=price_all,
        seed=CONFIG["seed"],
    )
    print("Best config (coarse search):", best_1)
    print(results_1.head(10))

    # Bước 6: grid search lần 2 (tinh chỉnh alpha, l1_ratio với window cố định)
    print("\nGrid search 2 - tinh chỉnh alpha và l1_ratio (score: price MSE 1-step):")
    results_2, best_2 = grid_search_alpha_l1(
        X_all_scaled=X_all_scaled,
        y_all=y_all,
        val_mask=val_mask,
        price_all=price_all,
        base_window_size=best_1["window_size"],
        base_window_type=best_1["window_type"],
        seed=CONFIG["seed"],
    )
    print("Best config (fine search):", best_2)
    print(results_2.head(10))

    # Bước 7: dùng cấu hình tốt nhất để rolling predict trên toàn bộ chuỗi
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

    print("\nKết quả 1-step-ahead return trên tập validation:")
    m0 = evaluate_predictions(y_true_val, y_pred_val)
    print(f"MSE (return): {m0['mse']:.8f}, MAE (return): {m0['mae']:.8f}, n_val_used: {mask_val_used.sum()}")

    # Bước 8: calibration tuyến tính y_true = a + b * y_pred
    print("\nCalibrating return predictions trên validation:")
    cal = LinearRegression()
    cal.fit(y_pred_val.reshape(-1, 1), y_true_val)
    y_pred_val_cal = cal.predict(y_pred_val.reshape(-1, 1))
    m_cal = evaluate_predictions(y_true_val, y_pred_val_cal)
    print(f"Calibration coefficients: a={cal.intercept_:.6e}, b={cal.coef_[0]:.6f}")
    print(f"After calibration MSE (return): {m_cal['mse']:.8f}, MAE (return): {m_cal['mae']:.8f}")

    # Bước 9: đánh giá theo price level
    idx_val_used = np.where(mask_val_used)[0]
    val_times = feat_df["time"].values[idx_val_used]

    time_to_close = dict(zip(df["time"].values, df["close"].values))

    price_t_arr = np.array([time_to_close.get(t, np.nan) for t in val_times], dtype=float)
    price_tp1_true_arr = price_t_arr * np.exp(y_true_val)

    price_pred_naive = price_t_arr.copy()
    price_pred_model = price_t_arr * np.exp(y_pred_val_cal)

    price_err_naive = mean_squared_error(price_tp1_true_arr, price_pred_naive)
    price_err_model = mean_squared_error(price_tp1_true_arr, price_pred_model)
    print("\nValidation 1 step PRICE MSE:")
    print(f"  Naive:  {price_err_naive:.6f}")
    print(f"  Model:  {price_err_model:.6f}")

    # Bước 10: ensemble đơn giản giữa naive và model trên price
    best_w = 0.0
    best_price_mse = np.inf
    for w in np.linspace(0.0, 1.0, 21):
        p_blend = w * price_pred_naive + (1.0 - w) * price_pred_model
        mse_blend = mean_squared_error(price_tp1_true_arr, p_blend)
        if mse_blend < best_price_mse:
            best_price_mse = mse_blend
            best_w = w

    print("\nBest ensemble weight trên price validation:")
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

    # Áp calibration cho chuỗi return tương lai
    future_returns_cal = cal.predict(future_returns_raw.reshape(-1, 1)).reshape(-1)

    # Chuyển thành path giá theo model
    last_price = df["close"].iloc[-1]
    price_future_model = returns_to_prices(
        last_price=last_price,
        future_returns=future_returns_cal,
    )

    # Path giá naive: luôn giữ nguyên giá cuối cùng
    price_future_naive = np.full_like(price_future_model, last_price, dtype=float)

    # Ensemble với trọng số đã tối ưu trên validation
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
