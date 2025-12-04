# patchtst/main.py

import numpy as np

from patchtst_best.config import (
    TARGET_COL,
    HORIZON,
    SMOOTH_RATIO,
    SMOOTH_METHOD,
)
from patchtst_best.data import (
    load_train_csv,
    load_test_csv,
    split_train_val,
    prepare_neuralforecast_frames,
    get_test_ground_truth,
)
from patchtst_best.patchtst_model import build_patchtst_model, make_neuralforecast
from patchtst_best.optuna_tuning import run_optuna
from patchtst_best.postprocessing import (
    collect_postprocessing_data,
    train_linear_post_model,
    apply_postprocessing,
)
from patchtst_best.metrics import compute_metrics, print_metrics
from patchtst_best.smooth_bias import (
    smooth_bias_correction_with_postprocessing,
    evaluate_smooth_method,
)


def main():
    # 1. Load dữ liệu
    df_train = load_train_csv()
    df_test = load_test_csv()

    train_data, val_data, _ = split_train_val(df_train, TARGET_COL)
    train_nf, val_nf, train_nf_full = prepare_neuralforecast_frames(
        df_train, train_data, val_data
    )

    y_true = get_test_ground_truth(df_train, df_test, HORIZON)

    # 2. Optuna tối ưu PatchTST
    best_params, best_mse_optuna, _ = run_optuna(train_nf_full)

    # 3. Train PatchTST baseline trên toàn bộ data
    print("\n================ Baseline PatchTST ================")
    model_baseline = build_patchtst_model(best_params, HORIZON)
    nf_baseline = make_neuralforecast(model_baseline)
    nf_baseline.fit(df=train_nf_full, val_size=0)
    forecast_baseline = nf_baseline.predict()
    pred_col = [c for c in forecast_baseline.columns if c not in ["unique_id", "ds"]][0]
    pred_baseline = forecast_baseline[pred_col].values[: len(y_true)]

    baseline_metrics = compute_metrics(y_true, pred_baseline)
    print_metrics("PatchTST Baseline", baseline_metrics)

    # 4. Train post processing regression (Linear Regression)
    print("\n================ Post processing ================")
    X_post, y_post = collect_postprocessing_data(train_nf_full, best_params)
    post_model = train_linear_post_model(X_post, y_post)
    pred_post, post_metrics, coef, intercept = apply_postprocessing(
        pred_baseline, post_model, y_true
    )
    print_metrics("Post processing", post_metrics, baseline_mse=baseline_metrics["mse"])

    # 5. Áp dụng Smooth Linear 20 percent
    print("\n================ Smooth Linear 20 percent (Best Method) ================")
    pred_smooth, weights = smooth_bias_correction_with_postprocessing(
        pred_baseline,
        post_model,
        method=SMOOTH_METHOD,
        smooth_ratio=SMOOTH_RATIO,
    )

    smooth_metrics = evaluate_smooth_method(
        pred_baseline,
        pred_smooth,
        post_model,
        y_true,
        baseline_mse=baseline_metrics["mse"],
        post_mse=post_metrics["mse"],
    )

    # 6. Placeholder: lưu submission nếu cần
    # Bạn có thể ghi thêm:
    # import pandas as pd
    # submission = pd.DataFrame({"y": pred_smooth})
    # submission.to_csv("submission_patchtst_best_method.csv", index=False)
    # và sau đó điền MSE test 100 ngày vào báo cáo.


if __name__ == "__main__":
    main()
