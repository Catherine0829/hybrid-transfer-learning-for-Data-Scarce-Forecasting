"""
Shared hyperparameters and paths for all experiments.
Edit this file to change settings; experiment scripts import from here.
"""

import os
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
# Directory containing <ticker>.csv and start_dates.xlsx
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# ── Train / test split ───────────────────────────────────────────────────────
SPLIT_DATE = pd.Timestamp('2024-01-01')

# ── Signature hyper-parameters ───────────────────────────────────────────────
LAGS   = [2, 3, 5, 10]
DEPTHS = [2, 3, 4]

# ── Regularisation ───────────────────────────────────────────────────────────
# Ridge–Ridge
RIDGE_LAMBDA_S = 0.0001
RIDGE_LAMBDA_T = 0.0005

# Lasso–Lasso
LASSO_LAMBDA_S = 1.0
LASSO_LAMBDA_T = 5.0

# Ridge source + Lasso transfer  (hybrid, main contribution of the paper)
HYBRID_LAMBDA_S = 0.0001
HYBRID_LAMBDA_T = 0.0005

# ── Wasserstein order ────────────────────────────────────────────────────────
P = 2

# ── Stocks for coefficient / visualisation analysis ──────────────────────────
TARGET_STOCKS = ['601127.SS', '9863.HK', 'GM', 'STLA']
