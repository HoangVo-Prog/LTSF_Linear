# LTSF_Linear


Competition URL:
https://www.kaggle.com/competitions/aio-2025-linear-forecasting-challenge

## Giải thích pipeline
Stage 0: Chuẩn hóa dữ liệu gốc → tạo return/volume change → winsorize outlier

Stage 1: Build feature mạnh nhất (lags, volatility, technical, trend…) + StandardScaler
Các nhóm feature: --> Đào sâu ý nghĩa của từng metrics. 

    Lags

    ret_lag1…ret_lag10

    vol_lag1…vol_lag5
    → Bắt độ nhớ ngắn hạn.

    Volatility + stats

    rolling std 5/10/20

    rolling min/max

    z-score 20 ngày
    → Bắt ức chế/đột biến theo biến động.

    SMA + price_trend

    SMA10, SMA20

    (price – SMA)/SMA
    → Bắt momentum.

    RSI

    → Bắt động lượng mua bán.

    Bollinger width

    → Biết giai đoạn giá “co” hay “mở rộng”.

    Calendar

    dow, month 
    → Chu kỳ tuần, tháng.

    Target y

    y = return ngày t+1

Stage 2: Grid search tìm window_size, window_type, alpha, l1_ratio tối ưu (theo MSE)

Stage 3: Tune lại alpha/l1_ratio nhưng cố định window_size + window_type từ Stage 2

Rolling ElasticNet dự báo return 1 bước (return 1-step)

Calibration: fit y_true = a + b y_pred trên năm 2025 để hiệu chỉnh bias của linear model (under-react hoặc over-react)

Ensemble price: P_final = w P_naive + (1 – w) P_model // dataset + tuning 

Forecast 100 ngày bằng ElasticNet + calibration + ensemble

---

Future works

1. Feature importance (correlation, ...) about feature engineering.  
Linear 2 tầng: Trend OLS + Residual ElasticNet/Ridge
2. Kalman Filter Regression
3. ARIMAX/VAR
4. OLS/DLinear/NLinear + tuning  
5. Transformer/LLM
    PatchTST (đặc tính chu kì?) / TimesNet / TSMixer / iTransformer
    TimesGPT, ...

8 months: train + random 
4 months: val 

tìm chu kỳ tốt nhất? FPT 
