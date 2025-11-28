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

    # Quantile để winsorize returns
    "clip_lower_q": 0.01,
    "clip_upper_q": 0.99,

    # Số ngày tương lai muốn dự đoán cho submission
    "forecast_steps": 100,

    # Số lần lặp tối đa cho ElasticNet
    "elastic_max_iter": 5000,
}

# Danh sách tên feature dùng để train model
FEATURE_NAMES: List[str] = [
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
    # Volatility của return với nhiều cửa sổ
    "vol_5",
    "vol_10",
    "vol_20",
    "vol_60",
    "vol_100",
    # Rolling stats trên return
    "ret_roll_min_20",
    "ret_roll_max_20",
    "ret_z_20",
    "mean_ret_5",
    "mean_ret_10",
    "mean_ret_20",
    "mean_ret_60",
    "mean_ret_100",
    # SMA và trend theo giá
    "sma10",
    "sma20",
    "sma50",
    "sma100",
    "price_trend_10",
    "price_trend_20",
    "price_trend_50",
    "price_trend_100",
    # Regime, drawdown dài hạn
    "drawdown_100",
    "days_since_max_100",
    # Chỉ báo kỹ thuật khác và lịch
    "rsi_14",
    "bb_width_20",
    "dow",
    "month",
]
