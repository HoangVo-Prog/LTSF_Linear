Ok, mình tổng hợp lại pipeline thành một bản mô tả chi tiết, rõ block, rồi vẽ sẵn bảng để bạn ghi kết quả submission từng model nhé.

---

## 1. Tổng quan Pipeline Direct 100d

**Mục tiêu**:
Dự báo **log return 100 ngày** cho FPT:

$$[
y_{direct}(t) = lp_{t+100} - lp_t
]$$

Sau đó chuyển sang **endpoint price**:

$$[
P_{t+100}^{pred} = \exp(lp_t + \hat{y}_{direct}(t))
]$$

Toàn bộ pipeline chia thành các block:

1. Load và chuẩn hóa dữ liệu gốc
2. Feature engineering
3. Xây target y_direct và time based folds
4. Feature selection nâng cao (ranking 4 chiều)
5. Định nghĩa các model base
6. Tuning hyperparameter trên CV MSE(endpoint price)
7. Train lại với best config và thu OOF
8. Stacking ensemble bằng Ridge trên OOF
9. Inference test + xuất submission cho từng model và ensemble

---

## 2. Block 1: Load data và tiền xử lý cơ bản

**Input**: file giá lịch sử FPT (OHLCV, ngày, v.v.)

Các bước:

* Đọc DataFrame với các cột cơ bản:
  `time, open, high, low, close, volume`
* Tính log price:

  * `lp = log(close)`
* Tính daily log return:

  * `ret_1d = lp.diff(1)`
* Tính log volume:

  * `vol_log = log(volume + 1)`
* Tạo phiên bản clipped:

  * `ret_1d_clip`, `vol_log_clip` để hạn chế outlier
* Đảm bảo sort theo `time` và reset index

Output Block 1: `df_base` với các cột gốc + lp + ret_1d + vol_log + clip.

---

## 3. Block 2: Feature engineering

Hàm `build_features(df)` sẽ:

### 3.1. Returns và volatility

* Multi horizon log returns:

  * `ret_2d, ret_3d, ret_5d, ret_10d, ret_20d, ret_30d, ret_60d, ret_120d`
* Volatility (rolling std của ret_1d):

  * `vol_5, vol_10, vol_20, vol_30, vol_60, vol_120`
* Volatility của log volume:

  * `vol_vol_20`

### 3.2. Moving average và relative price

* Simple moving average trên close:

  * `sma_5, sma_10, sma_20, sma_30, sma_60, sma_90, sma_120, sma_200`
* Exponential moving average:

  * `ema_5, ema_12, ema_20, ema_26, ema_50`
* Tỉ lệ giá so với SMA:

  * `price_sma5, price_sma10, price_sma20, price_sma60, price_sma90, price_sma120, price_sma200`

### 3.3. Drawdown và range

* Drawdown:

  * `dd_20, dd_60, dd_120, dd_200`
* Range intraday:

  * `hl_range, hl_range_abs` nếu có `high, low`

### 3.4. Volume features

* Moving average volume:

  * `vol_sma_5, vol_sma_10, vol_sma_20, vol_sma_60`
* Normalized volume:

  * `vol_norm_5, vol_norm_10, vol_norm_20, vol_norm_60`
* Z score volume:

  * `vol_z_20`
* Dollar volume:

  * `dollar_vol = close * volume`
* Tương quan 20 ngày giữa ret và vol:

  * `corr_ret_vol_20`

### 3.5. Technical indicators

* RSI:

  * `rsi_7, rsi_14, rsi_28`
* Bollinger Bands cho 20 và 60:

  * `bb_up_20, bb_low_20, bb_width_20`
  * `bb_up_60, bb_low_60, bb_width_60`
* MACD:

  * `macd, macd_signal, macd_hist`
* MACD nhanh:

  * `macd_fast, macd_fast_signal, macd_fast_hist`
* Stochastic, ATR, CCI nếu có high/low:

  * `stoch_k_14, stoch_d_14`
  * `atr_14`
  * `cci_20`

### 3.6. Calendar features

* `dow = day of week`
* `month`
* `month_sin, month_cos` (encoding seasonal)

### 3.7. Lag features

Tập lag base:

* `ret_1d_clip, vol_log_clip, ret_5d, ret_20d, ret_60d`
* `vol_5, vol_20, vol_60, vol_120`
* `price_sma*`
* `rsi_*, macd_*, macd_fast_*`
* `bb_width_20, bb_width_60, dd_20, dd_60, dd_120`
* `vol_norm_20, vol_z_20, stoch_k_14, stoch_d_14, cci_20, atr_14`
* `dollar_vol, corr_ret_vol_20`

Lags: 1, 2, 3, 5, 10, 20
Tạo cột `col_lagX` tương ứng.

Output Block 2: `df_feat` với toàn bộ feature + cột gốc.

---

## 4. Block 3: Target direct 100d và Time based CV folds

### 4.1. Target `y_direct`

Dùng hàm từ `targets_direct.py`:

* Với horizon H = 100:

  * `y_direct(t) = lp_{t+H} - lp_t`
* Căn chỉnh lại index để bỏ các dòng không có đủ 100 ngày phía trước

### 4.2. Time based folds

Dùng `splits.py`:

* Chia dữ liệu thành `n_folds` theo thời gian
* Mỗi fold có:

  * `train_mask`
  * `val_mask`
* Bảo đảm:

  * train < val theo time, không rò rỉ tương lai

Output Block 3:

* `df_direct` = df_feat có thêm `y_direct`
* `folds` = list folds với train/val mask

---

## 5. Block 4: Feature selection nâng cao

Hàm `run_feature_selection_direct` trong `feature_selection.py`:

### 5.1. Inputs

* `df_direct`
* `folds`
* `feature_cols` (các cột được coi là feature, loại bỏ time, target, id, v.v.)

### 5.2. Ranking 4 chiều

Cho mỗi feature:

1. **Mutual Information với y_direct**

   * Dùng `mutual_info_regression`
   * Xử lý NaN, inf, fill median
   * Mi_score càng cao → liên hệ phi tuyến càng mạnh

2. **Multi scale correlation với y_direct smoothed**

   * Tạo 5 phiên bản smoothed của y_direct:

     * EMA 5, EMA 20, EMA 60
     * Rolling mean 40, rolling median 50
   * Corr tuyệt đối giữa feature và từng series, rồi trung bình
   * Corr_score cao → feature đi cùng hướng với xu hướng 100d

3. **Stability Selection với ElasticNet trên folds**

   * Với mỗi fold:

     * Lấy train_mask
     * Fit ElasticNet (alpha nhỏ, l1_ratio 0.7) trên train
     * Đếm feature nào có coef ≠ 0
   * Stability_score = số lần xuất hiện / số model
   * Cao → feature ổn định, ít bị random

4. **Predictability Score (AR(1) R²)**

   * Với mỗi feature X:

     * Fit X_t ~ a * X_{t-1}
     * Predictability_score = R²
   * Cao → có cấu trúc, không phải pure noise trắng

### 5.3. Chuẩn hóa và trộn score

* Chuẩn hóa từng score về [0, 1]:

  * `mi_norm, corr_norm, stability_norm, predictability_norm`
* Final_score =

  * 0.3 * mi_norm
  * * 0.3 * corr_norm
  * * 0.3 * stability_norm
  * * 0.1 * predictability_norm
* Sort theo final_score giảm dần
* Chọn `top_k` (ví dụ 60 hoặc 80) feature

Output Block 4:

* `selected_features` (list)
* `rank_df` (DataFrame chi tiết score từng feature)

---

## 6. Block 5: Định nghĩa model base

Trong `models_direct.py`:

Các model chính:

* `DirectElasticNetModel`
* `DirectRidgeModel`
* `DirectXGBoostModel`
* `DirectLGBMModel`
* `DirectRandomForestModel`
* `DirectGBDTModel`

Mẫu logic:

* `fit(X_train, y_train)`
* `predict_100day_return(X)` trả về dự đoán `y_direct_hat`
* Wrapper chuẩn hóa (StandardScaler) cho X và y nếu cần
* Tree model (XGB, LGBM, RF, GBDT) dùng X_raw, target y_direct

---

## 7. Block 6: Tuning hyperparameter

`tnuning_direct.py` và `main_pipeline1_direct.py`:

### 7.1. Grid hyperparameter cho từng model

Ví dụ:

* ElasticNet:

  * alpha grid
  * l1_ratio grid
* Ridge:

  * alpha grid
* XGB:

  * n_estimators, max_depth, learning_rate, subsample, colsample_bytree, tree_method
* LGBM, RF, GBDT tương tự

### 7.2. Loop tuning

Cho mỗi `model_name` và mỗi `config` trong grid:

1. Tạo model
2. Với từng fold:

   * Lấy `train_mask` và `val_mask`
   * Train trên train
   * Predict y_direct_hat trên val
   * Convert thành giá endpoint:

     * `price_true, price_hat = compute_endpoint_price_from_direct(df_val, y_hat)`
   * Tính MSE(price)
3. Trung bình MSE trên tất cả folds → `cv_score` cho config đó

### 7.3. Chọn best config per model

* Giữ `best_configs[model_name]`
* Lưu JSON:

  * `model_params/best_params_all_models.json`

Output Block 6:

* `best_configs`
* `best_scores` (CV score per model)

---

## 8. Block 7: Train lại với best config và thu OOF

Sau tuning:

* Chạy lại CV với best_configs để:

  * Thu OOF predictions cho từng model
  * Thu OOF true price

Cho mỗi model:

1. Loop folds:

   * Train trên train_mask với best_config
   * Predict y_direct_hat trên val_mask
   * Convert sang price endpoint
   * Ghi lại vào mảng OOF
2. Kết quả:

   * `price_true_all`: vector true price trên toàn train
   * `price_hat_model`: vector OOF prediction của model

Ghép tất cả model:

* `price_hat_matrix` shape (N, M)
* M là số model base sử dụng

Output Block 7:

* `price_true_all`
* `price_hat_matrix`
* `used_models` (tên các base model)

---

## 9. Block 8: Stacking ensemble

Hàm `train_stacking_meta_learner` (Ridge hoặc ElasticNet):

* Input:

  * `oof_pred_matrix = price_hat_matrix`
  * `y_true = price_true_all`
* Fit Ridge:

  * `meta_model.fit(oof_pred_matrix, y_true)`
* In ra:

  * `coef_` cho từng base model
  * `intercept_`

Tại inference:

* Lấy prediction price của từng base model trên test
* Ghép thành matrix (shape: N_test x M)
* Meta_model.predict(matrix) → price ensemble

Output Block 8:

* `meta_model`
* `meta_model` coefficients per model (để interpret)

---

## 10. Block 9: Inference test và submission

### 10.1. Chuẩn bị dữ liệu test

* Load test CSV
* Build features bằng `build_features`
* Căn chỉnh sao cho đủ 100 ngày sau đó
* Chọn `selected_features` đã được FS

### 10.2. Predict từng base model

Đối với mỗi base model:

1. Train lại trên toàn bộ train (hoặc full data trừ last 100 nếu bạn muốn cẩn thận)
2. Predict y_direct_hat trên test
3. Convert thành price endpoint bằng `compute_endpoint_price_from_direct`
4. Lưu prediction:

* `pred_{model_name}`

### 10.3. Predict ensemble

* Tập hợp các `pred_{model_name}` thành matrix
* Meta_model.predict(matrix) → `pred_ensemble`

### 10.4. Tạo file submission

* Tùy format Kaggle, ví dụ:

  * `Id`, `Prediction`
* Bạn có thể tạo:

  * 1 file per base model
  * 1 file ensemble

---

## 11. Bảng kết quả submission

### 11.1. Các feature được chọn:

```python

```

### 11.2. Bảng kết quả từng model base

Bạn có thể dùng bảng này trong README hoặc notebook log:

```markdown
### Kết quả submission từng model (Pipeline 1 Direct 100d)

| Model           | Top_k features | CV MSE (price) | Public LB |
|----------------|----------------|----------------|-----------|
| elasticnet     | 100             |                |           |
| ridge          | 100             |                |           |
| xgboost        | 100             |                |           | 
| lgbm           | 100             |                |           |
| random_forest  | 100             |                |           | 
| gbdt           | 100             |                |           |  
```

Bạn chỉ cần điền:

* CV MSE (price) từ tuning
* Public LB, Private LB từ Kaggle
* Ghi chú đặc biệt (ví dụ model bị overfit, config khác seed)

---

### 11.3. Bảng kết quả ensemble

```markdown
### Kết quả ensemble

| Ensemble name     | Base models dùng                     | Meta model | CV MSE (price) | Public LB |
|-------------------|--------------------------------------|-----------|----------------|-----------|
| stack_ridge_60    | enet, ridge, xgb, lgbm, rf, gbdt    | Ridge     |                |           |
| stack_enet_60     | enet, ridge, xgb, lgbm, rf, gbdt    | ElasticNet|                |           |
| avg_top3          | xgb, lgbm, gbdt                     | mean      |                |           | 
```

---

