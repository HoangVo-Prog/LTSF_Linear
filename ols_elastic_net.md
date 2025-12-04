# MÔ TẢ ĐẦY ĐỦ PIPELINE DỰ BÁO GIÁ FPT 100 NGÀY BẰNG TREND + RESIDUAL MODEL

Toàn bộ pipeline gồm 6 khối chính:

1. **Cấu hình thí nghiệm và tiền xử lý dữ liệu gốc**
2. **Mô hình xu hướng dài hạn (TrendModel)**
3. **Xây dựng đặc trưng kỹ thuật và dataset supervised cho residual**
4. **Huấn luyện residual model, chọn đặc trưng lõi và tối ưu hyperparameter**
5. **Forecast nhiều bước (100 ngày) và kiểm tra trên validation**
6. **Dự báo 100 ngày cuối cùng và tạo submission**

Bên dưới là mô tả chi tiết từng khối, kèm vai trò, input, output và logic xử lý.

---

# 1. Cấu hình thí nghiệm và tiền xử lý dữ liệu gốc

## 1.1. Cấu hình chung trong `config.py`

Toàn bộ tham số thí nghiệm được tập trung trong `config.py`:

* Đường dẫn dữ liệu:

  * `TRAIN_CSV`: file chứa dữ liệu lịch sử FPT
  * `SUBMISSION_TEMPLATE_CSV`: file template submission của BTC
  * `OUTPUT_SUBMISSION_CSV`: tên file submission đầu ra

* Mốc thời gian chia tập:

  * `TRAIN_END_DATE`: mọi quan sát có `time < TRAIN_END_DATE` thuộc train, còn lại thuộc validation

* Cấu hình mô hình:

  * `TREND_POLY_DEGREE`: bậc đa thức của mô hình xu hướng (TrendModel)
  * `RESIDUAL_MODEL_TYPE`: loại model residual (`"elasticnet"` hoặc `"ridge"`)
  * `RESIDUAL_SHRINK`: hệ số thu nhỏ residual trong quá trình forecast nhiều bước

* Cấu hình feature:

  * `MAX_LAG`: số lag tối đa khi xây supervised dataset cho residual
  * `RET_WINDOWS`: các cửa sổ tính return (1, 5, 20 ngày)
  * `VOL_WINDOWS`: các cửa sổ tính rolling vol
  * `SMA_WINDOWS`: các cửa sổ SMA và các đặc trưng liên quan

* Cấu hình feature selection và search:

  * `TOP_K_FEATURES`: số lượng feature lõi giữ lại sau permutation importance
  * `N_RANDOM_SEARCH`: số trial random search cho hyperparameter residual model

* Cấu hình dự báo:

  * `FORECAST_STEPS`: số ngày cần dự báo cho submission (100) 

---

## 1.2. Load dữ liệu, chuẩn hóa cơ bản và log price

Hàm `load_price_data` trong `data_utils.py` thực hiện:

1. Đọc CSV `TRAIN_CSV`
2. Parse cột `time` sang kiểu datetime
3. Sắp xếp theo `time` và reset index
4. Tạo cột chỉ số thời gian rời rạc:

   * `t = index` dạng integer, dùng làm input cho TrendModel
5. Tính cột:

   * `log_price = log(close + 1e-8)`
     Đây là target chính để mô hình xu hướng làm việc, đồng thời cũng được dùng trong feature kỹ thuật. 

---

## 1.3. Chia train và validation theo mốc ngày

Hàm `train_val_split` chia dữ liệu theo `TRAIN_END_DATE`:

* `train_df`: mọi dòng có `time < TRAIN_END_DATE`
* `val_df`: mọi dòng có `time >= TRAIN_END_DATE` 

Trong `main.py`, đoạn đầu thực hiện:

```python
df = load_price_data(TRAIN_CSV)
train_df, val_df = train_val_split(df, TRAIN_END_DATE)
```

Sau đó in ra:

* Khoảng thời gian của train và validation
* Số lượng điểm trong mỗi phần

Đồng thời tính:

* `price_min_hist = df["close"].min()`
* `price_max_hist = df["close"].max()`

Và thiết lập biên clip cho log_price:

* `clip_low = log(0.9 * min_price_hist)`
* `clip_high = log(1.1 * max_price_hist)`

Biên này dùng xuyên suốt trong forecast để chặn log_price không trôi quá xa so với lịch sử. 

---

# 2. Mô hình xu hướng dài hạn (TrendModel)

## 2.1. Ý tưởng

Giá cổ phiếu thường có xu hướng dài hạn tương đối mượt, còn phần nhiễu ngắn hạn là biến động quanh xu hướng đó. Pipeline tách bài toán thành hai tầng:

1. Học xu hướng trơn của `log_price` theo thời gian bằng mô hình đa thức bậc thấp
2. Học residual (log_price trừ trend) bằng mô hình tuyến tính regularized (ElasticNet hoặc Ridge)

Điều này giúp residual model tập trung mô tả cấu trúc ngắn hạn, thay vì phải gánh luôn xu hướng dài hạn.

---

## 2.2. Cài đặt TrendModel

Lớp `TrendModel` trong `trend_model.py` gồm:

* `PolynomialFeatures(degree)` với `include_bias=True`
* `LinearRegression` để fit trên `log_price`

Cụ thể:

* Input: cột `t` (index thời gian)
* Biến đổi `t` sang ma trận đa thức `[1, t, t^2, ..., t^degree]`
* Fit LinearRegression trên `log_price` 

---

## 2.3. Thêm cột trend và residual vào DataFrame

Phương thức `add_trend_and_residual`:

1. Copy `df`
2. Dùng `predict_on_index(df["t"].values)` để ước lượng:

   * `trend_t = f(t)` trên toàn bộ lịch sử
3. Thêm hai cột:

   * `trend`
   * `resid = log_price - trend` 

Trong `main.py`, mô hình xu hướng được fit một lần trên **toàn bộ lịch sử**:

```python
trend_model = TrendModel(degree=TREND_POLY_DEGREE)
trend_model.fit(df)
df_trend_all = trend_model.add_trend_and_residual(df)
``` 

Đây là điểm quan trọng: xu hướng dài hạn được ước lượng bằng tất cả dữ liệu 2020 2025, sau đó dùng chung cho mọi phần sau.

---

# 3. Xây dựng đặc trưng kỹ thuật và dataset supervised cho residual

## 3.1. Feature engineering với `add_technical_features`

Hàm `add_technical_features` trong `features.py` nhận vào `df` đã có:

- `time`, `t`, `log_price`, `close`, `volume`, `resid`, v.v.

và tạo thêm hệ thống đặc trưng kỹ thuật, bao gồm:

1. **Return và log return cơ bản:**

   - `ret_1d = close.pct_change()`  
   - `log_ret_1d = diff(log_price)`  

   Và cho mỗi cửa sổ `w` trong `RET_WINDOWS`:

   - `log_ret_{w}d = log_price.diff(w)`  
   - `ret_{w}d = close.pct_change(w)`  

2. **Rolling volatility trên log_ret:**

   Với mỗi `w` trong `VOL_WINDOWS`:

   - `vol_{w}d = rolling_std(log_ret_1d, window=w)`  

3. **Simple Moving Average và tương quan với SMA:**

   Với mỗi `w` trong `SMA_WINDOWS`:

   - `sma_{w} = rolling_mean(close, window=w)`  
   - `price_sma_{w}_rel = close / sma_{w} - 1`  

4. **Volume features:**

   Cũng trên mỗi `w` trong `SMA_WINDOWS`:

   - `vol_ma_{w} = rolling_mean(volume, window=w)`  
   - `vol_rel_{w} = volume / vol_ma_{w} - 1`  

5. **RSI 14 phiên:**

   - Dùng `delta = close.diff()`  
   - Tách `up` và `down`, tính `roll_up = mean(up, window=14)` và `roll_down = mean(down, window=14)`  
   - `rs = roll_up / (roll_down + 1e-8)`  
   - `rsi_14 = 100 - 100 / (1 + rs)` 

Kết quả: DataFrame mở rộng `df_feat_all` chứa đầy đủ OHLCV, trend, resid và toàn bộ technical features.

---

## 3.2. Xây dựng dataset supervised cho residual

Hàm `make_supervised_residual_dataset` tạo ra dataset dạng supervised cho residual model:

- **Target:**  
  - `target_resid_next = resid.shift(-1)`  
  Tức là residual của **ngày tiếp theo**  

- **Base feature list:**  
  Bắt đầu từ các cột nền:

```text
  ["resid", "log_price", "ret_1d", "log_ret_1d",
   log_ret_{w}d, ret_{w}d,
   vol_{w}d,
   sma_{w}, price_sma_{w}_rel, vol_ma_{w}, vol_rel_{w},
   rsi_14]
```

trong đó `w` chạy qua RET_WINDOWS, VOL_WINDOWS, SMA_WINDOWS, nhưng chỉ giữ những cột thực sự tồn tại trong DataFrame. 

* **Tạo lag:**

  Với mỗi cột `col` trong `base_cols`, và mỗi `lag` từ `0` đến `MAX_LAG`, tạo feature:

  * `cname = f"{col}_lag{lag}" = df[col].shift(lag)`

  Tức là mỗi thông tin được nhìn qua nhiều lag thời gian, giúp model nắm động lực gần đây.

* **Kết quả cuối:**

  * `X`: ma trận feature gồm tất cả cột `{base_feature}_lag{0..MAX_LAG}`
  * `y`: series `target_resid_next`
  * `feature_names`: list tên cột tương ứng

Sau đó:

* Loại bỏ mọi dòng có NaN ở X hoặc y
* Trả về `(X, y, feature_names)` 

Trong `main.py`, pipeline xây supervised dataset cho **toàn bộ lịch sử**:

```python
df_trend_all = trend_model.add_trend_and_residual(df)
df_feat_all = add_technical_features(df_trend_all)
X_all, y_all, feature_names_raw = make_supervised_residual_dataset(df_feat_all)
```

Sau đó dùng cột `time` tương ứng với các index của `X_all` để tách ra supervised train và val:

* `train_mask = time < TRAIN_END_DATE`
* `val_mask = time >= TRAIN_END_DATE` 

---

# 4. Huấn luyện residual model, chọn đặc trưng lõi và tối ưu hyperparameter

## 4.1. ResidualModel và chuẩn hóa

`ResidualModel` là một wrapper che phủ:

* `ElasticNet` hoặc `Ridge` từ sklearn
* Một `StandardScaler` để chuẩn hóa X

Đặc điểm:

* Khi `fit(X, y)`:

  * Nếu `use_scaler=True`: fit scaler trên X, transform thành Xs
  * Fit model gốc trên Xs
* Khi `predict(X)`:

  * Transform X qua scaler (nếu có)
  * Predict trên Xs
* `score_mse(X, y)`: trả về MSE giữa y và y_pred

Model gốc được đặt trong `_base_model`, truy cập qua property `.model`. 

---

## 4.2. Fit base residual model để tính feature importance

Trong `main.py`:

1. Tách supervised train và val:

   ```python
   X_train = supervised_df.loc[train_mask, used_feature_names]
   y_train = supervised_df.loc[train_mask, "target"]
   X_val   = supervised_df.loc[val_mask, used_feature_names]
   y_val   = supervised_df.loc[val_mask, "target"]
   ```

2. Khởi tạo model nền:

   * Nếu `RESIDUAL_MODEL_TYPE == "elasticnet"`:

     * `alpha=0.01`, `l1_ratio=0.5`, `max_iter=10000`
   * Ngược lại dùng `Ridge(alpha=1.0)`

3. Fit trên `(X_train, y_train)` và tính MSE trên train và val để làm baseline. 

---

## 4.3. Permutation importance và chọn top K feature

Để model forecast 100 ngày ổn định hơn, ta không dùng toàn bộ feature mà chọn ra một tập đặc trưng lõi.

Pipeline:

1. Dùng `compute_feature_importance(base_model, X_val, y_val)`:

   * `permutation_importance` ở sklearn hoán vị từng cột feature nhiều lần
   * Đo mức giảm performance MSE, từ đó tính mean importance cho mỗi feature
   * Trả về `importances` dạng `np.ndarray` độ dài = số feature 

2. Lấy `feature_names_val = list(X_val.columns)`

3. Dùng `select_top_k_features(importances, feature_names_val, TOP_K_FEATURES)`:

   * Sort index `idx_sorted = argsort(importances)[::-1]`
   * Chọn `k` index đứng đầu
   * Trả về list `top_features` tương ứng

4. Loại bỏ mọi feature bắt đầu bằng `"resid_lag"`:

   ```python
   top_features = [f for f in top_features if not f.startswith("resid_lag")]
   ```

   Lý do: nếu dùng trực tiếp các lag của residual, forecast nhiều bước rất dễ tích lũy sai số và diverge.

5. Nếu sau khi loại resid_lag mà không còn feature nào:

   * Tìm tập index không phải residual
   * Chọn 20 feature không phải resid_lag có importance cao nhất làm fallback. 

---

## 4.4. Tối ưu hyperparameter bằng random search

Sau khi chọn core feature, pipeline:

1. Lọc lại X:

   ```python
   X_train_core = filter_feature_matrix(X_train, top_features)
   X_val_core   = filter_feature_matrix(X_val, top_features)
   ```

2. Nếu mô hình là ElasticNet:

   * Gọi `random_search_elasticnet(X_train_core, y_train, X_val_core, y_val, n_trials=N_RANDOM_SEARCH)`

   Trong mỗi trial:

   * Sample:

     * `alpha` từ log-uniform trong [1e−4, 1]
     * `l1_ratio` từ uniform trong [0, 1]
   * Fit ResidualModel với hyper này trên train
   * Tính `mse_val`
   * Cập nhật `best_model`, `best_params` nếu tốt hơn

3. Nếu mô hình là Ridge:

   * Gọi `random_search_ridge` với:

     * `alpha` trong [1e−4, 1e2] theo log-uniform

Kết quả: một `best_model` trên core features và dict `best_params`. 

---

## 4.5. Refit final residual model trên toàn bộ supervised data

Để sử dụng tối đa dữ liệu, sau random search, code:

1. Tạo `X_all_core = filter_feature_matrix(X_all, top_features)`

2. Lấy `y_all_core = y_all.loc[X_all_core.index]`

3. Khởi tạo `final_residual_model` với:

   * `model_type = best_model.model_type`
   * Các hyperparameter lấy từ `best_params`:

     * `alpha`, `l1_ratio` (nếu ElasticNet) hoặc `alpha` (nếu Ridge)

4. Fit `final_residual_model.fit(X_all_core, y_all_core)` trên toàn bộ supervised dataset. 

Kết quả: một residual model mạnh nhất có thể, dùng toàn bộ dữ liệu quá khứ và tập đặc trưng lõi ổn định.

---

# 5. Forecast nhiều bước (100 ngày) và kiểm tra trên validation

## 5.1. Hàm `forecast_future_prices`: recursive multi step forecast

Hàm này hiện thực logic dự báo nhiều bước trong `forecast.py`:

Input chính:

* `df_hist_raw`: lịch sử giá thực đến thời điểm bắt đầu forecast
* `trend_model`: TrendModel đã fit
* `residual_model`: final ResidualModel đã huấn luyện
* `feature_names`: danh sách top_features lõi
* `steps`: số bước dự báo (ví dụ 100)
* `log_clip_low`, `log_clip_high`: biên clip cho log_price
* `residual_shrink`: hệ số thu nhỏ residual (từ config) 

Quy trình cho mỗi bước forecast:

1. **Chuẩn hóa history:**

   * Sắp xếp `df_hist` theo `time` và reset index
   * Tạo lại `t = index`
   * Tính lại `log_price = log(close + 1e-8)`

2. **Thêm trend và residual, build feature:**

   * Dùng `trend_model.add_trend_and_residual(df_hist)` để tạo `trend` và `resid`
   * Gọi `add_technical_features` để thêm toàn bộ technical features

3. **Tạo supervised residual dataset:**

   * Gọi `make_supervised_residual_dataset(df_feat)`
   * Thu được `X_all`, `y_all`, `all_feat_names`

4. **Giữ lại core features:**

   * `X_all = filter_feature_matrix(X_all, feature_names)`

5. **Dự báo residual tiếp theo:**

   * Lấy dòng cuối cùng `x_latest = X_all.iloc[[-1]]`
   * `resid_pred = residual_model.predict(x_latest)`
   * `resid_next = residual_shrink * resid_pred`

6. **Tái dựng log_price kế tiếp:**

   * `last_log_price = df_hist["log_price"].iloc[-1]`
   * Trend anchor được chọn chính là giá trị log_price cuối cùng này
   * `log_price_next = last_log_price + resid_next`
   * Clip `log_price_next` vào `[log_clip_low, log_clip_high]`

7. **Chuyển về price và append vào history:**

   * `price_next = exp(log_price_next)`
   * Tạo một dòng mới với:

     * `time = last_time + 1 day`
     * `open = high = low = close = price_next`
     * `volume` và `symbol` được lấy từ dòng cuối cùng của history
   * Append vào `df_hist` và lưu `price_next` vào list `preds`

8. Lặp lại bước 1 7 `steps` lần. 

Kết quả: mảng `preds` chứa đường giá forecast đa bước.

---

## 5.2. Đánh giá path MSE 100 ngày trên validation

Để kiểm tra khả năng dự báo đa bước của pipeline, `main.py` có hàm:

```python
evaluate_path_mse_on_validation(
    df_full=df,
    trend_model=trend_model,
    residual_model=final_residual_model,
    feature_names=top_features,
    start_date="2024-01-02",
    horizon=100,
    log_clip_low=clip_low,
    log_clip_high=clip_high,
)
```

Logic của `evaluate_path_mse_on_validation`:

1. `df_hist = df_full[time <= start_date]` làm history  
2. `df_future_true = df_full[time > start_date]` sắp xếp theo thời gian, reset index  
3. Gọi `forecast_future_prices` với `steps = horizon` để dự báo 100 ngày tiếp theo từ history  
4. Lấy `true_prices = df_future_true["close"].iloc[:horizon]`  
5. Tính:

```python
   mse_path = mean((preds - true_prices)^2)
```

6. In ra:

   * Path MSE
   * Range giá thật vs range giá dự báo

Đây là nơi bạn sẽ ghi **MSE 100 ngày trên validation**.

---

# 6. Dự báo 100 ngày cuối cùng và tạo submission

## 6.1. Dự báo 100 ngày từ toàn bộ lịch sử

Sau khi đã tin tưởng pipeline, mô hình cuối cùng được dùng để dự báo tương lai thực tế cho submission:

```python
preds_future = forecast_future_prices(
    df_hist_raw=df,
    trend_model=trend_model,
    residual_model=final_residual_model,
    feature_names=top_features,
    steps=FORECAST_STEPS,
    log_clip_low=clip_low,
    log_clip_high=clip_high,
    residual_shrink=RESIDUAL_SHRINK,
)
```

Ở đây:

* `df_hist_raw` chính là full lịch sử FPT có sẵn trong train
* `FORECAST_STEPS = 100`
* Output `preds_future` là mảng 100 giá `close` dự báo cho 100 ngày tiếp theo. 

---

## 6.2. Ghép với template và xuất submission

Cuối cùng, pipeline đọc template submission:

1. Đọc `sub_template = pd.read_csv(SUBMISSION_TEMPLATE_CSV)`

2. Kiểm tra `len(sub_template) == FORECAST_STEPS`

3. Tạo bản copy:

   ```python
   submission = sub_template.copy()
   submission["close"] = preds_future.astype(float)
   submission.to_csv(OUTPUT_SUBMISSION_CSV, index=False)
   ```

4. In một vài dòng đầu để kiểm tra, và thông báo đường dẫn file submission. 

---

# PHẦN CUỐI: FORMAT CHO KẾT QUẢ SUBMISSION

Dưới đây là khung bạn có thể dùng trong report hoặc notebook để trình bày kết quả, bao gồm chỗ trống cho MSE 100 ngày.

---

## 🔹 1. File submission

File `submission.csv` gồm các cột giống template của BTC, trong đó cột `close` được thay bằng dự báo 100 ngày:

| id | close |
| ----- | -------------- |
| 0     | 116.051314     |
| 1     | 113.060426     |
| ...   | ...            |
| 99    | 17.667         |


---

## 🔹 2. Kết quả kiểm tra 100 ngày trên Leaderboard

```python
MSE: 5406.9571 
```
---

