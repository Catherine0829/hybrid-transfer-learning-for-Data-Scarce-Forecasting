"""
Data loading, path signature computation, and preprocessing utilities.

Key optimisation: precompute_signatures() pre-computes raw (un-standardised)
signatures once.  pre_processing_fast() then just slices + standardises for
each split date, avoiding repeated iisignature.sig() calls in the backtest.
"""

import numpy as np
import pandas as pd
import iisignature
from joblib import Parallel, delayed


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(file_path: str) -> pd.DataFrame:
    """Load OHLCV CSV and apply log transform to Close and Volume."""
    data = pd.read_csv(
        file_path,
        usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
    )
    data.index = pd.to_datetime(data['Date']).dt.normalize()
    data['Date']   = [t.toordinal() for t in data.index]
    close_mask = data['Close'] > 0
    volume_mask = data['Volume'] > 0
    data.loc[~close_mask, 'Close'] = np.nan
    data.loc[close_mask, 'Close'] = np.log(data.loc[close_mask, 'Close'])
    data.loc[~volume_mask, 'Volume'] = np.nan
    data.loc[volume_mask, 'Volume'] = np.log(data.loc[volume_mask, 'Volume'])
    return data.dropna()


# ── Signature features ────────────────────────────────────────────────────────

def compute_raw_signatures(data: pd.DataFrame, lag: int, depth: int) -> pd.DataFrame:
    """Return un-standardised path signatures for every rolling window."""
    values = data.values
    index = data.index[lag - 1:]
    sigs = [
        iisignature.sig(values[i: i + lag], depth)
        for i in range(len(values) - lag + 1)
    ]
    return pd.DataFrame(sigs, index=index)


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()


def compute_signature_features(data: pd.DataFrame, lag: int, depth: int) -> pd.DataFrame:
    """Compute and standardise path signature features."""
    return _standardize(compute_raw_signatures(data, lag, depth))


# ── Signature cache (for fast backtest) ──────────────────────────────────────

def precompute_signatures(
    data_frames: dict,
    start_dates: pd.DataFrame,
    lags: list,
    depths: list,
) -> dict:
    """
    Pre-compute raw signatures for every company and every (lag, depth) pair.

    Returns
    -------
    cache : dict
        cache[company][(lag, depth)] = pd.DataFrame of raw signatures
    """
    cache = {}
    for name in start_dates.index:
        data = data_frames[name].loc[start_dates.loc[name, 'Start date']:]
        cache[name] = {
            (lag, depth): compute_raw_signatures(data, lag, depth)
            for lag   in lags
            for depth in depths
        }
    return cache


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def _build_y(data: pd.DataFrame, lag: int, split_date):
    """Compute differenced, standardised target split into train/test."""
    y_raw = data['Close'].diff().iloc[lag - 1:]
    y_train  = y_raw.loc[:split_date]
    y_test  = y_raw.loc[split_date:]
    return (
        (y_train - y_train.mean()) / y_train.std(),
        (y_test - y_test.mean()) / y_test.std(),
    )


def _preprocess_single(
    name: str, start_date, raw_data: pd.DataFrame,
    lag: int, depth: int, split_date,
) -> tuple:
    data = raw_data.loc[start_date:]
    data_train = data.loc[:split_date]
    data_test = data.loc[split_date:]
    y_train, y_test = _build_y(data, lag, split_date)

    x_train = compute_signature_features(data_train, lag, depth).iloc[1:]
    x_test  = compute_signature_features(data_test, lag, depth)

    return name, {
        'x_train': x_train,
        'x_test':  x_test,
        'y_train': y_train.loc[x_train.index],
        'y_test':  y_test.loc[x_test.index],
    }


def _preprocess_from_cache_single(
    name: str, start_date, raw_data: pd.DataFrame,
    raw_sig: pd.DataFrame, lag: int, split_date,
) -> tuple:
    """Fast version: slice and re-standardise pre-computed signatures."""
    data = raw_data.loc[start_date:]
    y_train, y_test = _build_y(data, lag, split_date)
    split_pos = data.index.searchsorted(split_date, side='left')

    x_train = _standardize(raw_sig.loc[raw_sig.index <= split_date]).iloc[1:]
    if split_pos + lag - 1 < len(data.index):
        test_start = data.index[split_pos + lag - 1]
        x_test = _standardize(raw_sig.loc[raw_sig.index >= test_start])
    else:
        x_test = raw_sig.iloc[0:0].copy()

    return name, {
        'x_train': x_train,
        'x_test':  x_test,
        'y_train': y_train.loc[x_train.index],
        'y_test':  y_test.loc[x_test.index],
    }


# ── Public API ────────────────────────────────────────────────────────────────

def pre_processing(
    start_dates: pd.DataFrame, data_frames: dict,
    lag: int, depth: int, split_date,
) -> dict:
    """Full parallel preprocessing (computes signatures from scratch)."""
    results = Parallel(n_jobs=-1)(
        delayed(_preprocess_single)(
            name, start_dates.loc[name, 'Start date'],
            data_frames[name], lag, depth, split_date,
        )
        for name in start_dates.index
    )
    return dict(results)


def pre_processing_fast(
    start_dates: pd.DataFrame, data_frames: dict,
    sig_cache: dict, lag: int, depth: int, split_date,
) -> dict:
    """
    Fast parallel preprocessing using a pre-computed signature cache.
    Recommended for the rolling backtest where split_date changes monthly.
    """
    results = Parallel(n_jobs=-1)(
        delayed(_preprocess_from_cache_single)(
            name, start_dates.loc[name, 'Start date'],
            data_frames[name], sig_cache[name][(lag, depth)], lag, split_date,
        )
        for name in start_dates.index
    )
    return dict(results)
