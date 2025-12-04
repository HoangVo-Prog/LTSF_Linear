# patchtst/patchtst_model.py

from typing import Dict

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

from .config import HORIZON


def build_patchtst_model(
    params: Dict,
    horizon: int = HORIZON,
) -> PatchTST:
    """Tạo instance PatchTST từ dict params Optuna."""
    model = PatchTST(
        h=horizon,
        input_size=params["input_size"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        revin=True,
        learning_rate=params["learning_rate"],
        max_steps=params["max_steps"],
        val_check_steps=10,
    )
    return model


def make_neuralforecast(model: PatchTST) -> NeuralForecast:
    """Wrap model vào NeuralForecast."""
    nf = NeuralForecast(models=[model], freq="D")
    return nf
