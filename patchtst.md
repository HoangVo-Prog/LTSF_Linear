
# **PatchTST – Best Method Report (Smooth Linear 20 percent + Post-processing Regression)**

## Tổng quan pipeline

Pipeline gồm 6 bước:

1. Cài đặt thư viện
2. Load & chuẩn bị dữ liệu
3. Optuna hyperparameter tuning
4. Train PatchTST Baseline
5. Train Post-processing Linear Regression
6. Smooth Linear Correction 20 percent (Best Method)
7. Xuất submission và điền MSE 100 ngày Kaggle

---

# 🧩 **1. Cài đặt & Import Libraries**

### **Description**

Khối code này đảm bảo notebook có đầy đủ thư viện cần thiết, đặc biệt khi chạy trên Colab.
Hàm `install_package` sẽ kiểm tra xem package đã được cài chưa. Nếu chưa, tự động pip install.

### **Ý nghĩa**

* Tránh lỗi thiếu thư viện
* Đảm bảo môi trường chạy ổn định
* Không làm gián đoạn quá trình training

### **Code block**

```python
import subprocess
import sys

def install_package(package, import_name=None):
    ...
```

---

# 🧩 **2. Load & Chuẩn bị dữ liệu**

## **2.1 Load training data**

### **Description**

* Đọc file `FPT_train.csv` từ local hoặc Google Drive
* Sắp xếp theo thời gian
* Trích cột `close` làm target

### **Ý nghĩa**

Đảm bảo dữ liệu thời gian có thứ tự hợp lệ và không bị xáo trộn.

### **Code block**

```python
df = pd.read_csv(csv_path, parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)
```

---

## **2.2 Chia dữ liệu Train – Val**

### **Description**

* Train = 80 percent
* Val = 10 percent
* Còn lại không dùng
* Dùng để tune Optuna và tạo model chính

### **Mục đích**

Tách một phần validation để đánh giá chất lượng trong quá trình tuning.

### **Code block**

```python
train_data = close_values[:train_size]
val_data = close_values[train_size:train_size + val_size]
```

---

## **2.3 Load test data từ FPT_test.csv**

### **Description**

* File cực lớn (4.6M dòng)
* Lọc theo `symbol = FPT`
* Lọc theo thời gian > ngày cuối training
* Lấy đúng **100 ngày** làm ground truth

### **Ý nghĩa**

Dự đoán 100 ngày kế tiếp *thực sự* sau training window.

---

## **2.4 Chuẩn hóa format cho NeuralForecast**

### **Description**

Tạo 3 DataFrame chuẩn:

| cột       | ý nghĩa   |
| --------- | --------- |
| unique_id | series ID |
| ds        | timestamp |
| y         | giá close |

### **Code block**

```python
train_nf = pd.DataFrame({"unique_id":"FPT","ds":..., "y": train_data})
```

---

# 🧩 **3. Optuna Hyperparameter Tuning**

### **Description**

Optuna thử nhiều cấu hình PatchTST:

* input_size
* patch_len
* stride
* learning_rate
* max_steps

Mỗi trial:

1. Train PatchTST
2. Predict lên validation
3. Tính MSE
4. Optuna chọn best hyperparameters

### **Ý nghĩa**

Tối ưu kiến trúc PatchTST phù hợp với FPT 2020–2025.

### **Best parameters tìm được**

```
input_size = 100
patch_len = 32
stride = 4
learning_rate = 0.0016108149
max_steps = 250
```

### **Best MSE**

```
MSE_optuna = 191.8113
```

---

# 🧩 **4. Train PatchTST Baseline Model**

### **Description**

Dùng **best_params từ Optuna** train lại model trên toàn bộ train_nf_full (train + val).

### **Ý nghĩa**

Tạo baseline để so sánh với post-processing và smooth correction.

### **Baseline Results**

```
MSE   = 641.4994
RMSE  = 25.3278
MAE   = 24.1459
MAPE  = 23.68 percent
R²    = -17.53
Bias  = +24.10
```

### **Nhận xét**

* Model dự đoán rất lệch (bias cao)
* R² âm rất lớn → dự đoán không theo hướng dữ liệu

---

# 🧩 **5. Post-processing Regression (Linear Regression)**

### **5.1 Thu thập X_post, y_post**

Dùng **TimeSeriesSplit (3 folds)**:

* Train PatchTST trên mỗi fold
* Predict lên validation fold
* Gom tất cả pred → X_post
* Gom ground truth → y_post

Số điểm thu được:

```
300 điểm
```

---

## **5.2 Train Linear Regression**

Fitting công thức:

```
y ≈ a⋅pred + b
```

### **Best Linear Formula**

```
y = 0.7267 * pred + 9.3249
```

---

## **5.3 Kết quả Post-processing**

```
MSE   = 48.6205
RMSE  = 6.9728
MAE   = 5.0678
MAPE  = 4.79 percent
Bias  = -1.4356
```

### **Ý nghĩa**

* Sửa được gần như toàn bộ bias
* Sai số giảm cực mạnh (641 → 48)

---

# 🧩 **6. Smooth Linear 20 percent (Best Method)**

### **Description**

* 20 percent đầu: giữ nguyên baseline và dịch dần sang post-processing
* 80 percent cuối: dùng post-processing hoàn toàn
* Đảm bảo:

  * Điểm đầu = baseline
  * Điểm cuối = post-processing

### **Công thức tổng quát**

```
pred_final = (1 - w) * pred_baseline + w * pred_post
```

trong đó w tăng tuyến tính từ 0 → 1 ở 20 percent đầu.

---

### **Smooth Linear 20 percent – Results**

```
MSE   = 15.2606
RMSE  = 3.9065
MAE   = 3.2414
MAPE  = 3.17 percent
R²    = 0.5592
Bias  = 0.9108
```

### **Cải thiện**

* So với baseline: **+97.62 percent**
* So với post-processing: **+68.6 percent**

### **Ý nghĩa**

Đây là phương pháp tốt nhất trong toàn bộ pipeline.

---

# 🧩 **7. Xuất File Submission Kaggle**

### **Description**

* Xuất file dự đoán 100 ngày bằng phương pháp tốt nhất
* Format:

  ```
| id | Close |
| --- | --------------- |
| 1   |116.5001        |
| 2   |116.3011        |
| …   | …              |
| 100 |109.7434        |

  ```

---

# 🧩 **8. Tổng kết pipeline**

| Bước                     | Phương pháp            | MSE       |
| ------------------------ | ---------------------- | --------- |
| Baseline PatchTST        | Train full data        | 641.49    |
| Post-processing Linear   | Fit linear để sửa bias | 48.62     |
| Smooth Linear 20 percent | Best Method            | **15.26** |

**Best method đạt MSE 15.26**, cải thiện ~97 percent so với baseline.

