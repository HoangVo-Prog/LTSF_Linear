# patchtst/optuna_tuning.py

from typing import Dict, Tuple

import numpy as np
import optuna
from optuna import Trial
from sklearn.metrics import mean_squared_error

from .config import HORIZON, N_TRIALS
from .patchtst_model import build_patchtst_model, make_neuralforecast


def prepare_optuna_data(train_nf_full):
    """Cắt 90 percent train, 10 percent val phục vụ Optuna."""
    optuna_train_size = int(len(train_nf_full) * 0.9)
    train_nf_optuna = train_nf_full.iloc[:optuna_train_size].copy()
    val_nf_optuna = train_nf_full.iloc[optuna_train_size:].copy()

    val_close_optuna = val_nf_optuna["y"].values.astype("float32")
    print(" Dữ liệu cho Optuna:")
    print(f" - Train: {len(train_nf_optuna)}")
    print(f" - Val:   {len(val_nf_optuna)}")
    return train_nf_optuna, val_nf_optuna, val_close_optuna


def objective_patchtst(
    trial: Trial,
    train_nf_optuna,
    val_close_optuna: np.ndarray,
) -> float:
    """Objective function cho Optuna."""

    input_size = trial.suggest_int("input_size", 100, 300, step=50)
    patch_len = trial.suggest_int("patch_len", 8, 32, step=8)
    stride = trial.suggest_int("stride", 4, 16, step=4)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    max_steps = trial.suggest_int("max_steps", 50, 300, step=50)

    params = dict(
        input_size=input_size,
        patch_len=patch_len,
        stride=stride,
        learning_rate=learning_rate,
        max_steps=max_steps,
    )

    try:
        model = build_patchtst_model(params, horizon=min(HORIZON, len(val_close_optuna)))
        nf = make_neuralforecast(model)
        nf.fit(df=train_nf_optuna, val_size=0)
        forecast = nf.predict()

        pred_col = [c for c in forecast.columns if c not in ["unique_id", "ds"]][0]
        pred = forecast[pred_col].values

        n_points = min(len(pred), len(val_close_optuna), HORIZON)
        pred = pred[:n_points]
        val_true = val_close_optuna[:n_points]

        mse = mean_squared_error(val_true, pred)
        return mse
    except Exception as e:
        print(f" Lỗi trong trial: {e}")
        return float("inf")


def run_optuna(
    train_nf_full,
) -> Tuple[Dict, float, optuna.study.Study]:
    """Chạy Optuna và trả về best_params, best_mse, study."""
    train_nf_optuna, val_nf_optuna, val_close_optuna = prepare_optuna_data(
        train_nf_full
    )

    def _objective(trial: Trial) -> float:
        return objective_patchtst(trial, train_nf_optuna, val_close_optuna)

    study = optuna.create_study(
        direction="minimize", study_name="PatchTST_Optuna"
    )
    print(f" Đang tối ưu PatchTST với Optuna ({N_TRIALS} trials)")
    study.optimize(_objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_mse = study.best_value

    print(" Best parameters cho PatchTST:")
    for k, v in best_params.items():
        print(f" - {k}: {v}")
    print(f" - Best MSE: {best_mse:.4f}")

    return best_params, best_mse, study
