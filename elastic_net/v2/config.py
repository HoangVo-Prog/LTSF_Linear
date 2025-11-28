from typing import Dict, List

# =========================================
# Cấu hình chung cho toàn bộ pipeline
# =========================================

CONFIG: Dict[str, object] = {
    # Seed cho random, numpy
    "seed": 42,

    # Ngày bắt đầu tập validation
    # - Dữ liệu trước ngày này dùng để train
    # - Dữ liệu từ ngày này trở đi dùng để evaluate 1-step-ahead
    "val_start": "2025-01-01",

    # Quantile để winsorize returns làm feature input
    "clip_lower_q": 0.01,
    "clip_upper_q": 0.99,

    # Số ngày tương lai muốn dự đoán cho submission
    "forecast_steps": 100,

    # Số lần lặp tối đa cho ElasticNet
    "elastic_max_iter": 5000,

    # Tham số cho multi step validation
    # Horizon tối đa dùng để mô phỏng path trong validation
    "val_horizon": 50,
    # Số anchor trong đoạn validation để mô phỏng path
    "val_num_anchors": 5,
    # Ngưỡng an toàn cho slope calibration b, nếu vượt thì quay về identity
    "max_calib_slope": 3.0,
}

# Danh sách tên feature dùng để train model
# Lưu ý: đã bỏ hết feature dựa trên volume, thêm các feature chậm 60, 120, 252 ngày
FEATURE_NAMES: List[str] = [
    # Tín hiệu return ngắn hạn (winsorized)
    "ret_1d_clipped",
    "ret_lag1",
    "ret_lag2",
    "ret_lag3",
    "ret_lag4",
    "ret_lag5",
    "ret_lag6",
    "ret_lag7",
    "ret_lag8",
    "ret_lag9",
    "ret_lag10",

    # Volatility và thống kê rolling ngắn hạn của return
    "vol_5",
    "vol_10",
    "vol_20",
    "ret_roll_min_20",
    "ret_roll_max_20",
    "ret_z_20",
    "mean_ret_5",
    "mean_ret_10",
    "mean_ret_20",

    # Mức giá so với SMA
    "sma10",
    "sma20",
    "price_trend_10",
    "price_trend_20",

    # RSI, Bollinger width
    "rsi_14",
    "bb_width_20",

    # Feature chậm, mô tả regime 3 đến 12 tháng
    "cumret_60",
    "cumret_120",
    "cumret_252",
    "realized_vol_60",
    "realized_vol_120",
    "drawdown_60",
    "drawdown_120",
    "price_pct_252",

    # Feature theo lịch
    "dow",
    "month",
]
