import numpy as np
import pandas as pd
from config import HORIZON


def make_submission(price_path: np.ndarray, output_path: str = "submission.csv") -> None:
    """
    price_path: array length HORIZON chứa giá dự báo.
    """
    if len(price_path) != HORIZON:
        raise ValueError(f"Expected price_path length {HORIZON}, got {len(price_path)}")

    df_sub = pd.DataFrame(
        {
            "id": np.arange(1, HORIZON + 1),
            "close": price_path.astype(float),
        }
    )
    df_sub.to_csv(output_path, index=False)
