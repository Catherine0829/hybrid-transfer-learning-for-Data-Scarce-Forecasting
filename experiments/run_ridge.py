"""Run the Ridge transfer-learning experiment."""

import os
import sys

import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (  # noqa: E402
    DATA_DIR,
    DEPTHS,
    LAGS,
    P,
    RIDGE_LAMBDA_S as LAMBDA_S,
    RIDGE_LAMBDA_T as LAMBDA_T,
    SPLIT_DATE,
    TARGET_STOCKS,
)
from src.data import load_data, pre_processing  # noqa: E402
from src.evaluation import save_coefficients_excel, save_performance_excel  # noqa: E402
from src.transfer import pre_trained_ridge, ridge_transfer_coef, run_experiment  # noqa: E402


start_dates = pd.read_excel(
    os.path.join(DATA_DIR, "start_dates.xlsx"),
    usecols=["Code", "Start date"],
).dropna()
start_dates["Start date"] = pd.to_datetime(start_dates["Start date"])
start_dates.set_index("Code", inplace=True)

data_frames = {
    code: load_data(os.path.join(DATA_DIR, f"{code}.csv"))
    for code in start_dates.index
}


def _ridge_source_cutoff(frame, _name, absolute_idx):
    ts = frame["Start date"].iloc[absolute_idx - 11]
    year = ts.year if ts >= pd.Timestamp(f"{ts.year}-04-01") else ts.year - 1
    return pd.Timestamp(f"{year}-04-01")


def _run_one(lag, depth):
    pp = pre_processing(start_dates, data_frames, lag, depth, SPLIT_DATE)
    run_experiment(
        pp,
        start_dates,
        LAMBDA_S,
        LAMBDA_T,
        P,
        pre_trained_ridge,
        ridge_transfer_coef,
        Ridge,
        target_start_index=20,
        source_cutoff_fn=_ridge_source_cutoff,
    )
    return f"L={lag},M={depth}", pp


outputs = dict(
    Parallel(n_jobs=-1)(
        delayed(_run_one)(lag, depth)
        for lag in LAGS
        for depth in DEPTHS
    )
)

save_performance_excel(
    outputs,
    path=f"transfer_learning_performance_ridge_{LAMBDA_S}_{LAMBDA_T}.xlsx",
)
save_coefficients_excel(
    outputs,
    path=f"coef_ridge_{LAMBDA_S}_{LAMBDA_T}.xlsx",
    target_stocks=TARGET_STOCKS,
)
print("Done - Ridge experiment complete.")
