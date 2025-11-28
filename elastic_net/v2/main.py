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

    # Bước 3: build feature
    feat_df = build_features(df)

    # Flag train/val
    feat_df["is_val"] = feat_df["time"] >= pd.to_datetime(CONFIG["val_start"])

    X_all = feat_df[FEATURE_NAMES].values.astype(float)
    y_all = feat_df["y"].values.astype(float)
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

    # Bước 7: rolling forecast 1-step trên toàn bộ lịch sử để đánh giá
    preds_all = rolling_elasticnet_forecast(
        X=X_all_scaled,
        y=y_all,
        window_size=best_2["window_size"],
        alpha=best_2["alpha"],
        l1_ratio=best_2["l1_ratio"],
        window_type=best_2["window_type"],
        random_state=CONFIG["seed"],
    )

    # Chỉ dùng các điểm validation thực sự để đánh giá
    mask_val_used = val_mask & ~np.isnan(preds_all)
    y_true_val = y_all[mask_val_used]
    y_pred_val = preds_all[mask_val_used]


    print("\nKết quả 1-step-ahead return trên tập validation (log cho tham khảo):")
    m0 = evaluate_predictions(y_true_val, y_pred_val)
    print(f"MSE: {m0['mse']:.8f}, MAE: {m0['mae']:.8f}, n_val_used: {mask_val_used.sum()}")

    # ==========================
    # Calibration 1-step (fallback)
    # ==========================
    cal1 = LinearRegression()
    cal1.fit(y_pred_val.reshape(-1, 1), y_true_val)
    y_pred_val_cal1 = cal1.predict(y_pred_val.reshape(-1, 1)).reshape(-1)

    mse_before = mean_squared_error(y_true_val, y_pred_val)
    mse_after = mean_squared_error(y_true_val, y_pred_val_cal1)

    print("\nCalibration 1-step trên tập validation (fallback):")
    print(f"  a = {cal1.intercept_:.6e}, b = {cal1.coef_[0]:.6f}")
    print(f"  MSE trước calibration: {mse_before:.8f}")
    print(f"  MSE sau   calibration: {mse_after:.8f}")

    # 1-step price validation and ensemble (fallback)
    feat_val_idx = np.where(mask_val_used)[0]
    price_tp1_true_list = []
    price_pred_model_list = []
    price_pred_naive_list = []

    for idx_feat, y_pred_c in zip(feat_val_idx, y_pred_val_cal1):
        t_time = feat_df.loc[idx_feat, "time"]
        df_idx_arr = df.index[df["time"] == t_time]
        if len(df_idx_arr) == 0:
            continue
        df_idx = int(df_idx_arr[0])
        if df_idx + 1 >= len(df):
            continue

        price_t = float(df["close"].iloc[df_idx])
        price_tp1_true = float(df["close"].iloc[df_idx + 1])
        price_tp1_naive = price_t
        price_tp1_model = price_t * float(np.exp(y_pred_c))

        price_tp1_true_list.append(price_tp1_true)
        price_pred_naive_list.append(price_tp1_naive)
        price_pred_model_list.append(price_tp1_model)

    price_tp1_true_arr = np.array(price_tp1_true_list, dtype=float)
    price_pred_naive_arr = np.array(price_pred_naive_list, dtype=float)
    price_pred_model_arr = np.array(price_pred_model_list, dtype=float)

    price_err_naive = mean_squared_error(price_tp1_true_arr, price_pred_naive_arr)
    price_err_model = mean_squared_error(price_tp1_true_arr, price_pred_model_arr)

    print("\nValidation 1 step PRICE MSE (fallback):")
    print(f"  Naive:  {price_err_naive:.6f}")
    print(f"  Model:  {price_err_model:.6f}")

    best_w_1step = 0.0
    best_price_mse_1step = np.inf
    for w in np.linspace(0.0, 1.0, 21):
        p_blend = w * price_pred_naive_arr + (1.0 - w) * price_pred_model_arr
        mse_blend = mean_squared_error(price_tp1_true_arr, p_blend)
        if mse_blend < best_price_mse_1step:
            best_price_mse_1step = mse_blend
            best_w_1step = w

    print("\nBest ensemble weight trên price validation 1-step (fallback):")
    print(f"  w_naive = {best_w_1step:.2f}, w_model = {1.0 - best_w_1step:.2f}, MSE = {best_price_mse_1step:.6f}")

    # =========================================
    # Multi step validation mô phỏng pipeline submit
    # =========================================
    print("\nMulti step validation mô phỏng pipeline submit:")

    val_horizon_cfg = int(CONFIG.get("val_horizon", 50))
    val_num_anchors = int(CONFIG.get("val_num_anchors", 5))
    max_calib_slope = float(CONFIG.get("max_calib_slope", 3.0))

    times_feat = feat_df["time"].values
    is_val_feat = feat_df["is_val"].values
    val_idx_all = np.where(is_val_feat)[0]

    time_to_df_idx = {t: i for i, t in enumerate(df["time"].values)}

    # Tìm các index feature trong validation mà còn đủ dữ liệu tương lai
    candidate = []
    for idx_feat in val_idx_all:
        t = times_feat[idx_feat]
        df_idx = time_to_df_idx.get(t, None)
        if df_idx is None:
            continue
        max_future = len(df) - (df_idx + 1)
        if max_future >= 5:
            candidate.append((idx_feat, df_idx, max_future))

    if not candidate:
        print("  Không đủ dữ liệu để làm multi step validation, sẽ dùng 1-step fallback.")
        use_ms_calibration = False
        a_final = float(cal1.intercept_)
        b_final = float(cal1.coef_[0])
        best_w_final = best_w_1step
    else:
        max_future_global = max(mf for _, _, mf in candidate)
        val_horizon = min(val_horizon_cfg, max_future_global)
        print(f"  Horizon dùng cho multi step validation: {val_horizon}")

        # Chỉ chọn anchor còn đủ val_horizon
        valid_candidates = [(fi, di, mf) for (fi, di, mf) in candidate if mf >= val_horizon]
        if not valid_candidates:
            print("  Không có anchor nào đủ dài cho horizon này, dùng 1-step fallback.")
            use_ms_calibration = False
            a_final = float(cal1.intercept_)
            b_final = float(cal1.coef_[0])
            best_w_final = best_w_1step
        else:
            # Chọn tối đa val_num_anchors anchor trải đều
            feat_indices = [fi for (fi, _, _) in valid_candidates]
            if len(feat_indices) > val_num_anchors:
                anchor_pos = np.linspace(0, len(feat_indices) - 1, val_num_anchors).round().astype(int)
                anchor_feat_indices = [feat_indices[i] for i in anchor_pos]
            else:
                anchor_feat_indices = feat_indices

            print(f"  Số anchor dùng cho multi step validation: {len(anchor_feat_indices)}")

            all_pred_rets = []
            all_true_rets = []
            all_true_prices = []
            all_model_prices_uncal = []
            all_naive_prices = []
            anchor_last_prices = []

            for idx_feat in anchor_feat_indices:
                anchor_time = times_feat[idx_feat]
                df_idx = time_to_df_idx.get(anchor_time, None)
                if df_idx is None:
                    print("    Anchor không map được sang df, bỏ qua.")
                    continue

                print(f"    Anchor tại ngày: {anchor_time}")

                # Dữ liệu feature và target dùng để train static model tại anchor
                X_hist_scaled = X_all_scaled[: idx_feat + 1]
                y_hist = y_all[: idx_feat + 1]

                model_anchor = fit_final_elasticnet(
                    X_scaled=X_hist_scaled,
                    y=y_hist,
                    window_size=best_2["window_size"],
                    alpha=best_2["alpha"],
                    l1_ratio=best_2["l1_ratio"],
                    window_type=best_2["window_type"],
                    random_state=CONFIG["seed"],
                )

                df_hist = df.iloc[: df_idx + 1]
                max_future_here = min(val_horizon, len(df) - (df_idx + 1))
                if max_future_here < 5:
                    print("      Không đủ dữ liệu tương lai tại anchor này, bỏ qua.")
                    continue

                pred_rets = forecast_future_returns(
                    model=model_anchor,
                    scaler=scaler,
                    df=df_hist,
                    steps=max_future_here,
                )

                df_future = df.iloc[df_idx + 1 : df_idx + 1 + max_future_here]
                true_rets = df_future["ret_1d"].values.astype(float)
                true_prices = df_future["close"].values.astype(float)

                if len(true_rets) != len(pred_rets):
                    print("      Độ dài true và pred mismatch, bỏ qua anchor này.")
                    continue

                last_price = float(df_hist["close"].iloc[-1])
                price_true_path = returns_to_prices(last_price, true_rets)
                price_model_uncal_path = returns_to_prices(last_price, pred_rets)
                price_naive_path = np.full_like(price_true_path, last_price, dtype=float)

                all_pred_rets.append(pred_rets)
                all_true_rets.append(true_rets)
                all_true_prices.append(price_true_path)
                all_model_prices_uncal.append(price_model_uncal_path)
                all_naive_prices.append(price_naive_path)
                anchor_last_prices.append(last_price)

            if len(all_pred_rets) == 0:
                print("  Không có anchor hợp lệ, dùng 1-step fallback.")
                use_ms_calibration = False
                a_final = float(cal1.intercept_)
                b_final = float(cal1.coef_[0])
                best_w_final = best_w_1step
            else:
                # Flatten returns để fit calibration trong regime static multi step
                y_pred_flat = np.concatenate(all_pred_rets)
                y_true_flat = np.concatenate(all_true_rets)

                cal_ms = LinearRegression()
                cal_ms.fit(y_pred_flat.reshape(-1, 1), y_true_flat)
                a_ms = float(cal_ms.intercept_)
                b_ms = float(cal_ms.coef_[0])

                print("\nCalibration multi step (trên returns):")
                print(f"  a_ms = {a_ms:.6e}, b_ms = {b_ms:.6f}")

                # Guard chống slope quá lớn
                if abs(b_ms) > max_calib_slope:
                    print("  Slope quá lớn, dùng identity calibration.")
                    a_ms, b_ms = 0.0, 1.0

                # Đánh giá PRICE MSE multi step
                prices_true_flat = np.concatenate(all_true_prices)
                prices_naive_flat = np.concatenate(all_naive_prices)

                all_model_prices_cal = []
                for preds_anchor, last_price in zip(all_pred_rets, anchor_last_prices):
                    preds_cal = a_ms + b_ms * preds_anchor
                    price_model_cal_path = returns_to_prices(last_price, preds_cal)
                    all_model_prices_cal.append(price_model_cal_path)
                prices_model_flat = np.concatenate(all_model_prices_cal)

                mse_price_naive = mean_squared_error(prices_true_flat, prices_naive_flat)
                mse_price_model = mean_squared_error(prices_true_flat, prices_model_flat)

                print("\nValidation multi step PRICE MSE:")
                print(f"  Naive:  {mse_price_naive:.6f}")
                print(f"  Model:  {mse_price_model:.6f}")

                # Tìm best_w trên multi step price path
                best_w_ms = 0.0
                best_price_mse_ms = np.inf
                for w in np.linspace(0.0, 1.0, 21):
                    p_blend = w * prices_naive_flat + (1.0 - w) * prices_model_flat
                    mse_blend = mean_squared_error(prices_true_flat, p_blend)
                    if mse_blend < best_price_mse_ms:
                        best_price_mse_ms = mse_blend
                        best_w_ms = w

                print("\nBest ensemble weight trên multi step price validation:")
                print(f"  w_naive = {best_w_ms:.2f}, w_model = {1.0 - best_w_ms:.2f}, MSE = {best_price_mse_ms:.6f}")

                # Nếu multi step tốt hơn naive thì dùng calibration multi step, ngược lại fallback 1-step
                if mse_price_model < mse_price_naive:
                    use_ms_calibration = True
                    a_final, b_final = a_ms, b_ms
                    best_w_final = best_w_ms
                else:
                    print("  Multi step model không thắng naive, fallback về 1-step calibration.")
                    use_ms_calibration = False
                    a_final = float(cal1.intercept_)
                    b_final = float(cal1.coef_[0])
                    best_w_final = best_w_1step

    print("\nCalibration và ensemble dùng cho forecast 100 ngày:")
    print(f"  a_final = {a_final:.6e}, b_final = {b_final:.6f}, w_naive = {best_w_final:.2f}, w_model = {1.0 - best_w_final:.2f}")

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

    # Áp calibration cuối cùng
    future_returns_cal = a_final + b_final * future_returns_raw

    # Chuyển thành path giá
    last_price = float(df["close"].iloc[-1])
    price_future_model = returns_to_prices(last_price, future_returns_cal)
    price_future_naive = np.full_like(price_future_model, last_price, dtype=float)

    price_future_final = best_w_final * price_future_naive + (1.0 - best_w_final) * price_future_model

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
