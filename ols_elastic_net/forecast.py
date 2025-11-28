import numpy as np
import pandas as pd

from features import add_technical_features
from data_utils import make_supervised_residual_dataset
from feature_selection import filter_feature_matrix


def forecast_future_prices(
    df_hist_raw,
    trend_model,
    residual_model,
    feature_names,
    steps: int,
    log_clip_low: float,
    log_clip_high: float,
    residual_shrink: float = 0.7,
):
    """
    Recursive multi step forecast.

    At each step:
      1) Rebuild trend, residual and features on full current history
      2) Build supervised X, y (y ignored)
      3) Take last feature row and predict next residual
      4) Shrink residual and combine with trend at next index to get next log price
      5) Clip log price to [log_clip_low, log_clip_high]
      6) Convert to price, append to history and continue
    """
    df_hist = df_hist_raw.copy()
    preds = []

    for _ in range(steps):
        # Sort and recompute time index and log price
        df_hist = df_hist.sort_values("time").reset_index(drop=True)
        df_hist["t"] = df_hist.index.astype(int)
        df_hist["log_price"] = np.log(df_hist["close"].astype(float) + 1e-8)

        # 1. Add trend and residual, then technical features
        df_trend = trend_model.add_trend_and_residual(df_hist)
        df_feat = add_technical_features(df_trend)

        # 2. Build supervised residual dataset
        X_all, y_all, all_feat_names = make_supervised_residual_dataset(df_feat)

        # 3. Keep only selected features
        X_all = filter_feature_matrix(X_all, feature_names)

        # 4. Use last row to predict next residual
        x_latest = X_all.iloc[[-1]]
        resid_pred = float(residual_model.predict(x_latest)[0])

        # Apply shrink factor so residual impact does not explode over 100 steps
        resid_next = residual_shrink * resid_pred

        # 5. Trend for next index
        t_next = len(df_hist)
        trend_next = float(trend_model.predict_on_index(np.array([t_next]))[0])

        # 6. Combine trend and residual in log space and clip
        log_price_next = trend_next + resid_next
        log_price_next = float(
            np.clip(log_price_next, log_clip_low, log_clip_high)
        )

        # 7. Back to price
        price_next = float(np.exp(log_price_next))

        # 8. Next date is last time plus 1 day
        next_time = df_hist["time"].iloc[-1] + pd.Timedelta(days=1)

        new_row = {
            "time": next_time,
            "open": price_next,
            "high": price_next,
            "low": price_next,
            "close": price_next,
            "volume": df_hist["volume"].iloc[-1],
            "symbol": df_hist["symbol"].iloc[-1],
        }
        df_hist = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)
        preds.append(price_next)

    return np.array(preds, dtype=float)
