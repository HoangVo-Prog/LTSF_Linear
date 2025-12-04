#### Best result: 52.9284

# FPT Stock 100 Day Forecasting Pipeline

This repository implements an end to end pipeline to forecast a 100 day price path for the FPT stock using a linear ElasticNet model on engineered technical features built from price and volume data. The training objective is one step ahead log return prediction, while the final goal is a multi step price path forecast created by recursively rolling the model forward.

The project is organized into modular files that separate configuration, data utilities, feature engineering, model training, and forecasting logic.

---

## Project structure

* `config.py`
  Global configuration for the entire pipeline. Contains the validation start date, winsorization quantiles, forecast horizon, ElasticNet solver settings, and the ordered list of feature names used by both training and forecasting. 

* `data_utils.py`
  Utility functions for reproducibility, dataset discovery in Kaggle like environments, base return computation, and winsorization of returns and volume changes. 

* `features.py`
  Feature engineering logic. Builds a feature matrix from raw time series that already have base returns and clipped returns, and constructs the target variable `y` for supervised learning. 

* `models.py`
  All ElasticNet related utilities. Contains rolling one step ahead evaluation, final model fitting, prediction metrics, and a two stage grid search over window configuration and regularization parameters. 

* `forecasting.py`
  Multi step forecasting logic. Implements an autoregressive loop that rolls the trained ElasticNet model forward for 100 days using synthetic features built from a small history buffer of returns, volume changes, and prices. Also converts return paths to price paths. 

* `main.py`
  Orchestration script. Wires all modules together: loading data, preprocessing, feature building, scaling, grid search, validation, calibration, price level evaluation, simple ensemble with a naive baseline, final model fitting, and 100 day forecast generation and saving to `submission.csv`. 

* `FPT_train.csv`
  Training dataset with at least these columns: `time`, `close`, `volume`. Used as the only data source for both training and building the starting point of the 100 day forecast.

* `sample_submission.csv`
  Template for the required submission format for the competition. It has columns `id` and `close`, where `id` runs from 1 to 100 and `close` is the forecasted price for that step in the future.

---

## Configuration

All global settings live in `config.py`. Key keys in the `CONFIG` dictionary: 

* `seed`
  Random seed for `random` and `numpy` to make runs reproducible.

* `val_start`
  Date string (for example `"2025-01-01"`) that splits the time series into train and validation parts:

  * Dates before `val_start` are used as training history.
  * Dates on or after `val_start` are used for one step ahead validation.

* `clip_lower_q`, `clip_upper_q`
  Lower and upper quantiles used to winsorize (clip) the return and volume change distributions. These are estimated only on the training period, then applied to the entire series.

* `forecast_steps`
  Number of future business days to forecast. Set to 100, matching the competition requirement.

* `elastic_max_iter`
  Maximum number of iterations for the ElasticNet solver.

* `FEATURE_NAMES`
  Ordered list of all feature names. Used in both training and forecasting to guarantee consistent column order between the training design matrix and the on the fly feature vector in the autoregressive loop.

---

## Data utilities and preprocessing

Implemented in `data_utils.py`. 

### Reproducibility and file loading

* `set_seed(seed: int)`
  Sets seeds for Python `random` and `numpy` to ensure reproducible results.

* `find_data_path(filename: str) -> Path`
  Looks for a given file in typical Kaggle input locations, then falls back to the current directory. This allows the same code to run both in Kaggle and locally without modification.

### Base return and volume change computation

* `add_base_returns(df: pd.DataFrame) -> pd.DataFrame`
  Takes a dataframe with at least `close` and `volume` columns and adds:

  * `close_shift1`: previous close.
  * `volume_shift1`: previous volume.
  * `ret_1d`: one day log return of price `log(close / close_shift1)`.
  * `vol_chg`: one day log change of volume `log((volume + 1) / (volume_shift1 + 1))`. The `+1` avoids log of zero when volume is zero.

### Winsorization of returns

* `winsorize_returns(df, val_start, lower_q, upper_q)`
  Clips `ret_1d` and `vol_chg` to reduce the influence of extreme outliers:

  * Only dates before `val_start` are used to estimate quantiles.
  * Creates `ret_1d_clipped` and `vol_chg_clipped` as clipped variants.

Internally this uses `clip_by_quantiles`, which:

* Filters to finite values inside the training mask.
* Computes `low` and `high` as the given quantiles.
* Clips the full series between `low` and `high`.

---

## Feature engineering

Feature engineering is implemented in `features.py` via `build_features`, with support functions for RSI. 

### RSI computation

* `_rsi_from_window(prices_window: np.ndarray, period: int)`
  Computes RSI from a single window of prices using the standard gain and loss based formula.

* `compute_rsi_series(close: pd.Series, period: int = 14) -> pd.Series`
  Rolls RSI across the entire close price series. For each index from `period` onward, it takes the last `period + 1` prices and calls `_rsi_from_window`. Returns a series aligned with the input.

### Full feature matrix and supervised target

* `build_features(df: pd.DataFrame) -> pd.DataFrame`

Expected input columns:

* `time`
* `close`
* `ret_1d_clipped`
* `vol_chg_clipped`

Output:

* `time`
* All feature columns listed in `FEATURE_NAMES`
* `y`: the target one day ahead return

Feature groups:

1. Return lags

   * `ret_lag1` to `ret_lag10` built from `ret_1d_clipped`.
   * `ret_lag1` is the current clipped return, `ret_lag2` is return from one day ago relative to the current row, and so on.

2. Volume change lags

   * `vol_lag1` to `vol_lag5` built from `vol_chg_clipped`.
   * Similar lag structure to return lags.

3. Return volatility features

   * `vol_5`, `vol_10`, `vol_20`: rolling standard deviation of `ret_1d_clipped` over 5, 10, and 20 days.

4. Return extremum and distribution features

   * `ret_roll_min_20`, `ret_roll_max_20`: rolling minimum and maximum of returns over the last 20 days.
   * `ret_z_20`: z score of current return relative to the last 20 days (rolling mean and rolling std).

5. Short and medium horizon mean returns

   * `mean_ret_5`, `mean_ret_10`, `mean_ret_20`: rolling means of `ret_1d_clipped` over 5, 10, 20 days.

6. Price level and trend features

   * `sma10`, `sma20`: simple moving averages of `close` over 10 and 20 days.
   * `price_trend_10`, `price_trend_20`: relative distance of current price to its SMA, `(close - smaX) / smaX`.

7. Momentum oscillator and band features

   * `rsi_14`: 14 day RSI on `close`.
   * `bb_width_20`: Bollinger band width over 20 days, computed as `(upper20 - lower20) / sma20` where the band is plus or minus 2 standard deviations around SMA20.

8. Calendar features

   * `dow`: day of week.
   * `month`: month of year.

Target:

* `y = ret_1d_clipped.shift(-1)`
  This is the clipped log return at day t+1, so the model learns to predict next day return using information available up to day t.

The function drops rows with insufficient rolling history to ensure no missing values remain in the final feature matrix.

---

## ElasticNet model utilities

Implemented in `models.py`. 

### Model creation

* `_make_elasticnet(alpha, l1_ratio, random_state)`
  Creates an `ElasticNet` instance with:

  * Given regularization parameters `alpha` and `l1_ratio`.
  * Intercept enabled.
  * Deterministic seed.
  * Maximum iterations set from `CONFIG["elastic_max_iter"]`.

### Rolling one step ahead forecast for validation

* `rolling_elasticnet_forecast(X, y, window_size, alpha, l1_ratio, window_type="sliding", random_state=42)`

Core idea:

* Loop over time indices from `window_size` to `n_samples - 1`.
* For each index `i`, choose a training window:

  * `sliding` window: use `[i - window_size, i)` as training slice.
  * `expanding` window: use `[0, i)` as training slice.
* Check that `y_train` has at least two distinct values (otherwise skip to avoid degenerate fits).
* Fit a new ElasticNet on that window then predict `y[i]` from `X[i]`.
* Return an array of the same length as `y`, filled with `nan` where training was not possible.

This simulates a realistic one step ahead process that only uses past data at each prediction point.

### Final model fitting

* `fit_final_elasticnet(X_scaled, y, window_size, alpha, l1_ratio, window_type="sliding", random_state=42)`

Fits the final ElasticNet that will be used for future forecasting:

* If `window_type` is `"sliding"`, only the last `window_size` rows are used.
* If `window_type` is `"expanding"`, the full dataset is used.

This mirrors the best performing window strategy from validation but fits it once on all available data.

### Evaluation metrics

* `evaluate_predictions(y_true, y_pred)`
  Computes MSE and MAE for a pair of prediction and target arrays.

### Grid search over window and regularization

Two level search:

1. Coarse search

   * `grid_search_window_and_reg(X_all_scaled, y_all, val_mask, seed=42)`
   * Internally calls `_grid_search_elasticnet_mse` with:

     * `window_sizes` in `[126, 252, 504, 756]`.
     * `window_types` in `["sliding", "expanding"]`.
     * `alphas` and `l1_ratios` covering a coarse grid.
   * For each configuration:

     * Calls `rolling_elasticnet_forecast`.
     * Uses only validation indices where predictions are not `nan` (via `val_mask`).
     * Computes MSE and MAE, records number of validation points used.
   * Returns a sorted results dataframe and the best configuration as a dictionary.

2. Fine search given best window

   * `grid_search_alpha_l1(X_all_scaled, y_all, val_mask, base_window_size, base_window_type, seed=42)`
   * Uses the same rolling evaluation but only varies `alpha` and `l1_ratio` around a finer grid while keeping window size and type fixed to the coarse best.

---

## Multi step forecasting utilities

Implemented in `forecasting.py`. 

### Autoregressive multi step return forecast

* `forecast_future_returns(model, scaler, df, steps) -> np.ndarray`

Workflow:

1. Extract the non missing history of `ret_1d_clipped`, `vol_chg_clipped`, `close`, `time` from `df`.

2. Initialize buffers:

   * Last 20 returns.
   * Last 5 volume changes.
   * Last 20 prices.
   * Current date as the last observed date.

3. For each forecast step from 1 to `steps`:

   * Build a feature dictionary `feat_vals` using only the buffers, mirroring the training features:

     * Current `ret_1d_clipped` and `vol_chg_clipped`.
     * Return lags and volume lags filled from the buffer, defaulting to 0 when not enough history.
     * Rolling volatility, min, max, mean returns with the same windows as in training.
     * SMA10, SMA20 on prices and price trend features.
     * RSI 14 on price buffer, with sensible defaults when history is too short.
     * Bollinger width 20 computed from price buffer and SMA20.
     * Calendar features `dow` and `month` derived from `current_date`.
   * Assemble the feature vector in the exact `FEATURE_NAMES` order, scale with the fitted `StandardScaler`, and predict the next day return using the trained ElasticNet.
   * Append the predicted return to a prediction list and update buffers:

     * Update price buffer with `next_price = current_price * exp(r_pred_next)`.
     * Append predicted return to `ret_buffer`.
     * Append a placeholder `0.0` to volume buffer (volume future is not modeled and is set to zero change).
     * Increment `current_date` by one business day.

4. Return the array of predicted log returns for `steps` future days.

Note: after the first step, the buffers become a mix of real historical data and synthetic model outputs, which is a standard autoregressive setup.

### Return to price conversion

* `returns_to_prices(last_price, future_returns) -> np.ndarray`

Starting from `last_price`, iteratively multiplies by `exp(r_t)` for each predicted return to get the full future price path.

---

## Main pipeline flow

The full flow is orchestrated in `main.py`. 

### 1. Setup and data loading

* Calls `set_seed` for reproducibility.
* Uses `find_data_path("FPT_train.csv")` to locate and load the training dataset.
* Parses `time` as datetime, sorts the dataframe by `time`.

### 2. Base returns and winsorization

* Uses `add_base_returns` to create `ret_1d` and `vol_chg`.
* Calls `winsorize_returns` with:

  * `val_start` from `CONFIG`.
  * `clip_lower_q` and `clip_upper_q` from `CONFIG`.
* The dataframe now contains `ret_1d_clipped` and `vol_chg_clipped`.

### 3. Feature construction and split masks

* Calls `build_features` to generate the full feature matrix and target `y`.
* Creates a boolean column `is_val` to mark rows with `time >= val_start`.
* Extracts:

  * `X_all` as a numpy array of columns listed in `FEATURE_NAMES`.
  * `y_all` as the target vector.
  * `val_mask` and `train_mask` from `is_val`.

### 4. Feature scaling

* Fits `StandardScaler` on `X_all[train_mask]` only (training distribution).
* Applies the scaler to all rows to get `X_all_scaled`.

### 5. Hyperparameter tuning

* Runs `grid_search_window_and_reg` to get a coarse best `window_size`, `window_type`, `alpha`, and `l1_ratio`.
* Runs `grid_search_alpha_l1` with the coarse window settings to refine `alpha` and `l1_ratio`.

### 6. Rolling validation and calibration

* Calls `rolling_elasticnet_forecast` with the best configuration to obtain one step ahead predictions over the whole series.
* Uses only the validation portion with finite predictions to compute MSE and MAE via `evaluate_predictions`.
* Performs linear calibration of return predictions on the validation set:

  * Fits `LinearRegression` on `(y_pred_val, y_true_val)` pairs.
  * Uses the fitted intercept and slope to adjust return predictions.
* Evaluates one step ahead performance in price space:

  * Maps each validation target to its corresponding price level via `time` mapping.
  * Computes true next day prices from log returns.
  * Compares:

    * Naive predictor that keeps price unchanged.
    * Model predictor obtained from calibrated returns.
  * Computes MSE in price space for both.

### 7. Simple price level ensemble

* Searches for the best convex combination on price between:

  * Naive price forecast (constant last price).
  * Model price forecast (from calibrated returns).
* Loops over weights `w` from 0.0 to 1.0 and selects the one that minimizes validation price MSE.
* Stores the best `w` as `w_naive`, with `1 - w` for the model.

### 8. Final model fitting

* Uses `fit_final_elasticnet` with:

  * All scaled features `X_all_scaled`.
  * All targets `y_all`.
  * Best hyperparameters from grid search.
* This gives `final_model` that will be used for multi step forecasting.

### 9. Multi step 100 day forecast

* Calls `forecast_future_returns` with:

  * `final_model`.
  * Fitted `scaler`.
  * The full raw dataframe `df`.
  * `steps = CONFIG["forecast_steps"]` (100).
* Applies the same linear calibration to the multi step predicted returns:

  * `future_returns_cal = cal.predict(future_returns_raw.reshape(-1, 1))`.

### 10. Price path generation and ensemble

* Gets `last_price` as the last close from the dataset.

* Calls `returns_to_prices(last_price, future_returns_cal)` to create `price_future_model`, the model based price path.

* Creates `price_future_naive` as a flat path equal to `last_price` for all 100 days.

* Applies the previously optimized ensemble weight:

  * `price_future_final = w_naive * price_future_naive + (1 - w_naive) * price_future_model`.

* Builds a diagnostic dataframe `future_signals_df` with:

  * `step`, `ret_raw`, `ret_cal`, `price_model`, `price_naive`, `price_final`.

### 11. Submission file

* Creates `submission` dataframe with:

  * `id`: from 1 to `forecast_steps`.
  * `close`: `price_future_final` as float.
* Saves to `submission.csv` in the current working directory.

---

## How to run

1. Install dependencies (example using pip):

   ```bash
   pip install numpy pandas scikit-learn
   ```

2. Place `FPT_train.csv` in one of these locations:

   * Kaggle style input folder, for example `/kaggle/input/aio-2025-linear-forecasting-challenge`.
   * `../input/aio-2025-linear-forecasting-challenge`.
   * Current directory.

3. Run the main script:

   ```bash
   python main.py
   ```

4. After the script finishes, check:

   * Console logs for validation metrics and debug information.
   * Generated `submission.csv` file with the final 100 day price forecast.

---

## Extending or modifying the pipeline

* To add or change features:
  Modify `build_features` in `features.py` and update `FEATURE_NAMES` in `config.py` to keep order consistent.

* To change the validation scheme:
  Adjust `CONFIG["val_start"]` or extend `models.py` to support other validation masks.

* To tweak regularization search:
  Edit the grids in `grid_search_window_and_reg` and `grid_search_alpha_l1` in `models.py`.

* To explore other ensemble strategies:
  Replace the simple linear blend in `main.py` with more advanced combinations or nonlinear calibrations on price or return space.

This README should give you a full birds eye view of the flow from raw data to final 100 day price path forecast and help you navigate and extend the codebase.
