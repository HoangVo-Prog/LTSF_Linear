# patchtst_best/config.py

from pathlib import Path

# Google Drive IDs cho file dữ liệu
FPT_TRAIN_DRIVE_ID = "1nS9xshut38SJEX__PD_zjKFtj2CQCn7S"
FPT_TEST_DRIVE_ID = "1IkzoSTHPMnOUBILN7cCPjVw9QWAuOtCs"

# Đường dẫn mặc định
DATA_DIR = Path(".")
TRAIN_CSV_PATH = DATA_DIR / "FPT_train.csv"
TEST_CSV_PATH = DATA_DIR / "FPT_test.csv"

# Thiết lập forecast
TARGET_COL = "close"
HORIZON = 100

# Tỉ lệ chia train  val
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1  # phần còn lại làm test hoặc ground truth từ file riêng

# Optuna
N_TRIALS = 20

# Smooth bias correction
SMOOTH_RATIO = 0.2  # 20 percent đầu smooth
SMOOTH_METHOD = "linear"
