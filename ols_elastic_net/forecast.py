# forecast.py

from typing import List
import numpy as np
import pandas as pd

from trend_model import TrendModel
from residual_model import ResidualModel
from features import add_technical_features, make_supervised_residual_dataset


def forecast_future_prices(
    df_hist_raw: pd.DataFrame,
    trend_model: TrendModel,
    residual_model: ResidualModel,
    feature_names: List[str],
    steps: int,
) -> np.ndarray:
    """
    Recursive 100 step forecast.
    At each step:
      1) Rebuild trend, residual and features on full current history
      2) Build supervised X, y (y ignored)
      3) Take last feature row and predict next residual
      4) Combine with trend at next index to get next log price, then price
      5) Append predicted row to history and continue
    """
    df_hist = df_hist_raw.copy()
    preds = []

    for _ in range(steps):
        df_hist = df_hist.sort_values("time").reset_index(drop=True)
        df_hist["t"] = df_hist.index.astype(int)
        df_hist["log_price"] = np.log(df_hist["close"].astype(float) + 1e-8)

        # trend and resid
        df_trend = trend_model.add_trend_and_residual(df_hist)
        df_feat = add_technical_features(df_trend)

        X_all, y_all, all_feat_names = make_supervised_residual_dataset(df_feat)

        # keep only selected features
        from feature_selection import filter_feature_matrix

        X_all = filter_feature_matrix(X_all, feature_names)

        # use last row to predict next residual
        x_latest = X_all.iloc[[-1]]
        resid_next = residual_model.predict(x_latest)[0]

        # time index and trend for next step
        t_next = len(df_hist)
        trend_next = trend_model.predict_on_index(np.array([t_next]))[0]

        log_price_next = trend_next + resid_next

        # Old clipping
        # log_price_next = float(
        #     np.clip(log_price_next, np.log(1.0), np.log(1e4))
        # )

        # New, tighter clipping around realistic FPT range
        log_price_next = float(
            np.clip(log_price_next, np.log(10.0), np.log(300.0))
        )

        price_next = float(np.exp(log_price_next))


        # next date is last time plus 1 day
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

    return np.array(preds)
