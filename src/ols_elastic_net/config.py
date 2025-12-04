# config.py

TRAIN_CSV = "LTSF_Linear/data/aio-2025-linear-forecasting-challenge/FPT_train.csv"
SUBMISSION_TEMPLATE_CSV = "LTSF_Linear/data/aio-2025-linear-forecasting-challenge/sample_submission.csv"
OUTPUT_SUBMISSION_CSV = "submission.csv"

# Time split
TRAIN_END_DATE = "2023-01-01"  # train: < this date, validation: >= this date

# Two stage model configuration
TREND_POLY_DEGREE = 3

RESIDUAL_MODEL_TYPE = "elasticnet"  # "elasticnet" or "ridge"
RESIDUAL_SHRINK = 0.7  # 0.5 to 0.8 is a reasonable range

# Lags and windows for features
MAX_LAG = 5
RET_WINDOWS = [1, 5, 20]
VOL_WINDOWS = [20, 60]
SMA_WINDOWS = [20, 60, 120]

# Feature selection
TOP_K_FEATURES = 40  # number of core features to keep

# Optimization search settings
N_RANDOM_SEARCH = 25  # random hyperparameter tries

# Forecast
FORECAST_STEPS = 100
