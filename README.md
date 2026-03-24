# Hybrid Transfer Learning for EV Stock Return Prediction

Reference implementation for the ICAIF paper on EV stock return prediction with
path signatures and transfer learning.

## Overview

This repository predicts daily log-return changes for EV-related stocks using
path signature features built from OHLCV windows. Earlier-listed stocks are
used as source tasks and later-listed stocks are treated as target tasks.

Three experiment entry points are included:

| Script | Source model | Transfer penalty | Notes |
| --- | --- | --- | --- |
| `experiments/run_ridge.py` | Ridge | L2 | Ridge transfer experiment |
| `experiments/run_lasso.py` | Lasso | L1 | Lasso transfer experiment |
| `experiments/run_ridge_lasso.py` | Ridge | L1 | Hybrid setting from the paper |

Transfer risk is reported with a Wasserstein-based measure computed from the
source prediction and target test distribution.

## Research Setup

The project follows the EV-sector forecasting setup described in the paper and
presentation materials:

- Universe: 24 listed EV-related firms across the U.S., Mainland China, and
  Hong Kong markets.
- Start date per stock: the later of IPO date and January 1 of the year before
  the firm's first EV release.
- Main evaluation split: training through `2023-12-31`, testing on
  `2024-01-01` to `2024-12-31`.
- Source/target idea: earlier-listed firms provide source-task information for
  later-listed firms with shorter histories.
- Feature grid used in experiments: lag length `L in {2, 3, 5, 10}` and
  signature depth `M in {2, 3, 4}`.

## Method Sketch

For each `(lag, depth)` pair:

1. Pre-processing
   Build path signature features from rolling OHLCV windows.
   Build the target series from differenced log-close.
   Standardize train and test splits separately.
2. Source pre-training
   Fit a Ridge or Lasso source model on pooled source-company training data.
3. Target adaptation
   Optimize the transfer-learning objective with `scipy.optimize.minimize`.
4. Evaluation
   Save direct-learning metrics, transfer-learning metrics, transfer risk, and
   coefficient vectors.

## Repository Structure

```text
hybrid-transfer-learning/
|-- src/
|   |-- data.py
|   |-- transfer.py
|   |-- evaluation.py
|   `-- __init__.py
|-- experiments/
|   |-- config.py
|   |-- run_ridge.py
|   |-- run_lasso.py
|   `-- run_ridge_lasso.py
|-- backtest/
|   |-- strategy.py
|   `-- __init__.py
|-- data/
|   `-- README.md
|-- requirements.txt
`-- .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

`iisignature` may require a compiler toolchain. On Windows, using a binary
wheel is usually the easiest option:

```bash
pip install iisignature --prefer-binary
```

## Data Setup

1. Put `start_dates.xlsx` in `data/`.
2. Put one CSV per ticker in `data/`, for example `GM.csv`, `STLA.csv`.
3. Make sure the files match the format described in
   [data/README.md](data/README.md).

For local testing and benchmarking, this repository also includes
`data/sample(start_dates).xlsx`, which matches the 24-stock sample used in the
research workflow.

## Running Experiments

From the project root:

```bash
python experiments/run_ridge.py
python experiments/run_lasso.py
python experiments/run_ridge_lasso.py
```

Shared paths and hyperparameters live in
[experiments/config.py](experiments/config.py).

Excel outputs are written to the project root.

## Backtest

`backtest/strategy.py` provides a rolling walk-forward backtest utility built
around the same preprocessing and transfer-learning pipeline.

Example:

```python
import pandas as pd
from sklearn.linear_model import Ridge

from src.data import load_data, precompute_signatures
from src.transfer import pre_trained_ridge, lasso_transfer_coef
from backtest.strategy import (
    run_backtest,
    build_prediction_df,
    add_prediction_intervals,
    plot_strategy,
)

sig_cache = precompute_signatures(data_frames, start_dates, lags=[2], depths=[2])
date_range = pd.date_range("2024-01-01", "2024-11-30", freq="MS")

backtest = run_backtest(
    start_dates,
    data_frames,
    sig_cache,
    date_range,
    lag=2,
    depth=2,
    lambda_S=0.0001,
    lambda_T=0.0005,
    p=2,
    source_trainer=pre_trained_ridge,
    target_adapter=lasso_transfer_coef,
    direct_model_cls=Ridge,
)

graph = build_prediction_df(backtest, "GM", date_range)
graph = add_prediction_intervals(graph, date_range)
plot_strategy("GM", data_frames, graph, date_range)
```

## Benchmark

Using the 24-stock sample in `sample(start_dates).xlsx` together with the
corresponding daily CSV files, the current implementation was benchmarked on a
1-year walk-forward backtest over `2024-01-01` to `2024-12-01` with monthly
retraining, `lag=2`, and `depth=2`.


### 1-Year Backtest Metrics

Hybrid Ridge-Lasso strategy results for the 15 target stocks in the sample:

| Stock | Cumulative Return | Annualized Return | Max Drawdown | Sharpe |
| --- | ---: | ---: | ---: | ---: |
| `9863.HK` | `3.180767` | `3.490749` | `-0.250657` | `2.343497` |
| `601127.SS` | `2.362377` | `2.591637` | `-0.161152` | `2.768230` |
| `STLA` | `0.803396` | `0.811924` | `-0.095952` | `2.100313` |
| `GM` | `0.511238` | `0.516238` | `-0.098250` | `1.947248` |


## Notes

- The refactor keeps the modular project structure, parallel preprocessing,
  and reusable experiment runners.

## Environment

The benchmark above was run in a dedicated Conda environment on Windows 11 with:

- Python `3.14.3`
- `numpy 2.4.3`
- `pandas 3.0.1`
- `scipy 1.17.1`
- `scikit-learn 1.8.0`
- `joblib 1.5.3`
- `matplotlib 3.10.8`
- `iisignature 0.24`
- `openpyxl 3.1.5`

Additional document-processing packages used while preparing the repository:

- `pypdf 6.9.2`
- `python-pptx 1.0.2`

## Dependencies

- `iisignature`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `joblib`
- `matplotlib`
