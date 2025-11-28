import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

from config import FEATURE_NAMES


# =========================================
# Forecast multi step với autoregressive
# =========================================

def forecast_future_returns(
    model: ElasticNet,
    scaler: StandardScaler,
    df: pd.DataFrame,
    steps: int,
) -> np.ndarray:
    """
    Dự đoán nhiều bước tương lai (multi step) theo kiểu autoregressive.

    Ý tưởng:
      - Lấy lịch sử return và giá (tối đa 252 ngày gần nhất)
      - Mỗi bước:
        + Từ buffer hiện tại build 1 vector feature
        + Scale và cho model dự đoán return ngày tiếp theo
        + Cập nhật buffer bằng return dự đoán và giá mới
      - Lặp lại steps lần

    Lưu ý:
      - Từ bước thứ 2 trở đi, buffer đã trộn giữa dữ liệu thật và dự đoán.
    """
    non_na = df["ret_1d_clipped"].notna()
    ret_series = df.loc[non_na, "ret_1d_clipped"].values.astype(float)
    close_series = df.loc[non_na, "close"].values.astype(float)
    time_series = df.loc[non_na, "time"].values

    if len(ret_series) < 20:
        raise ValueError("Not enough history to forecast")

    # Dùng tối đa 252 ngày lịch sử để có thể tính slow regime features
    max_hist = 252
    ret_buffer = list(ret_series[-max_hist:])
    price_buffer = list(close_series[-max_hist:])
    current_date = pd.Timestamp(time_series[-1])

    preds = []

    def last_window(a: np.ndarray, w: int) -> np.ndarray:
        """Lấy w phần tử cuối, nếu không đủ thì lấy hết."""
        if len(a) >= w:
            return a[-w:]
        return a

    def rolling_std(a: np.ndarray, w: int) -> float:
        """Std của cửa sổ cuối cùng."""
        win = last_window(a, w)
        if len(win) <= 1:
            return 0.0
        return float(np.std(win, ddof=1))

    def rolling_mean(a: np.ndarray, w: int) -> float:
        """Mean của cửa sổ cuối cùng."""
        win = last_window(a, w)
        if len(win) == 0:
            return 0.0
        return float(np.mean(win))

    def rolling_sum(a: np.ndarray, w: int) -> float:
        """Sum của cửa sổ cuối cùng."""
        win = last_window(a, w)
        if len(win) == 0:
            return 0.0
        return float(np.sum(win))

    for _ in range(steps):
        feat_vals = {}

        ret_arr = np.array(ret_buffer, dtype=float)
        price_arr = np.array(price_buffer, dtype=float)

        current_ret = ret_arr[-1]
        current_price = price_arr[-1]

        # Các feature cơ bản
        feat_vals["ret_1d_clipped"] = current_ret

        # Return lags
        for k in range(1, 11):
            if len(ret_buffer) >= k:
                feat_vals[f"ret_lag{k}"] = ret_buffer[-k]
            else:
                feat_vals[f"ret_lag{k}"] = 0.0

        # Volatility và rolling stats (short horizon)
        feat_vals["vol_5"] = rolling_std(ret_arr, 5)
        feat_vals["vol_10"] = rolling_std(ret_arr, 10)
        feat_vals["vol_20"] = rolling_std(ret_arr, 20)

        feat_vals["ret_roll_min_20"] = float(np.min(last_window(ret_arr, 20)))
        feat_vals["ret_roll_max_20"] = float(np.max(last_window(ret_arr, 20)))
        mean20 = rolling_mean(ret_arr, 20)
        std20 = rolling_std(ret_arr, 20)
        feat_vals["mean_ret_20"] = mean20
        feat_vals["mean_ret_5"] = rolling_mean(ret_arr, 5)
        feat_vals["mean_ret_10"] = rolling_mean(ret_arr, 10)
        feat_vals["ret_z_20"] = (current_ret - mean20) / std20 if std20 != 0 else 0.0

        # SMA và trend từ buffer giá
        win10_price = last_window(price_arr, 10)
        win20_price = last_window(price_arr, 20)
        sma10_price = float(np.mean(win10_price)) if len(win10_price) > 0 else current_price
        sma20_price = float(np.mean(win20_price)) if len(win20_price) > 0 else current_price

        feat_vals["sma10"] = sma10_price
        feat_vals["sma20"] = sma20_price
        feat_vals["price_trend_10"] = (current_price - sma10_price) / sma10_price if sma10_price != 0 else 0.0
        feat_vals["price_trend_20"] = (current_price - sma20_price) / sma20_price if sma20_price != 0 else 0.0

        # RSI 14 từ chuỗi giá trong buffer
        win14_price = last_window(price_arr, 14)
        if len(win14_price) < 2:
            rsi_14 = 50.0
        else:
            delta = np.diff(win14_price)
            gain = np.where(delta > 0, delta, 0.0)
            loss = np.where(delta < 0, -delta, 0.0)
            avg_gain = gain.mean()
            avg_loss = loss.mean()
            if avg_loss == 0:
                rsi_14 = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_14 = 100.0 - 100.0 / (1.0 + rs)
        feat_vals["rsi_14"] = rsi_14

        # Bollinger width 20
        if len(win20_price) > 1:
            std20_price = float(np.std(win20_price, ddof=1))
            bb_width = (4.0 * std20_price) / sma20_price if sma20_price != 0 else 0.0
        else:
            bb_width = 0.0
        feat_vals["bb_width_20"] = bb_width

        # ---------- Slow regime features ----------

        # Cumulative log returns trên 60, 120, 252 ngày
        feat_vals["cumret_60"] = rolling_sum(ret_arr, 60)
        feat_vals["cumret_120"] = rolling_sum(ret_arr, 120)
        feat_vals["cumret_252"] = rolling_sum(ret_arr, 252)

        # Realized volatility 60, 120 ngày
        feat_vals["realized_vol_60"] = rolling_std(ret_arr, 60)
        feat_vals["realized_vol_120"] = rolling_std(ret_arr, 120)

        # Drawdown hiện tại so với đỉnh 60, 120 ngày
        win60_price = last_window(price_arr, 60)
        win120_price = last_window(price_arr, 120)
        if len(win60_price) > 0:
            max60 = float(np.max(win60_price))
            feat_vals["drawdown_60"] = current_price / max60 - 1.0 if max60 != 0 else 0.0
        else:
            feat_vals["drawdown_60"] = 0.0

        if len(win120_price) > 0:
            max120 = float(np.max(win120_price))
            feat_vals["drawdown_120"] = current_price / max120 - 1.0 if max120 != 0 else 0.0
        else:
            feat_vals["drawdown_120"] = 0.0

        # Price percentile trong 252 ngày
        win252_price = last_window(price_arr, 252)
        if len(win252_price) > 0:
            min252 = float(np.min(win252_price))
            max252 = float(np.max(win252_price))
            if max252 != min252:
                feat_vals["price_pct_252"] = (current_price - min252) / (max252 - min252)
            else:
                feat_vals["price_pct_252"] = 0.5
        else:
            feat_vals["price_pct_252"] = 0.5

        # Calendar feature cho ngày tiếp theo (business day)
        next_date = current_date + pd.offsets.BDay(1)
        feat_vals["dow"] = next_date.dayofweek
        feat_vals["month"] = next_date.month

        # Sắp xếp feature theo đúng thứ tự FEATURE_NAMES
        x_vec = np.array([feat_vals[name] for name in FEATURE_NAMES], dtype=float).reshape(1, -1)
        x_scaled = scaler.transform(x_vec)

        # Dự đoán return ngày tiếp theo
        r_pred_next = float(model.predict(x_scaled)[0])
        preds.append(r_pred_next)

        # Cập nhật buffer cho bước tiếp theo
        next_price = current_price * np.exp(r_pred_next)

        ret_buffer.append(r_pred_next)
        if len(ret_buffer) > max_hist:
            ret_buffer = ret_buffer[-max_hist:]

        price_buffer.append(next_price)
        if len(price_buffer) > max_hist:
            price_buffer = price_buffer[-max_hist:]

        current_date = next_date

    return np.array(preds, dtype=float)

# =========================================
# Chuyển chuỗi return thành chuỗi giá
# =========================================

def returns_to_prices(
    last_price: float,
    future_returns: np.ndarray,
) -> np.ndarray:
    """
    Từ giá hiện tại và chuỗi return tương lai, suy ra path giá tương lai.
    """
    prices = []
    p = float(last_price)
    for r in future_returns:
        p = p * np.exp(r)
        prices.append(p)
    return np.array(prices, dtype=float)
