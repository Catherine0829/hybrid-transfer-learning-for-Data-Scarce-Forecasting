"""
Transfer learning core: source pre-training, target adaptation, and the
parallel experiment runner.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance
from joblib import Parallel, delayed


# ── Source pre-training ───────────────────────────────────────────────────────

def pre_trained_ridge(X: np.ndarray, y: np.ndarray, lambda_S: float) -> np.ndarray:
    """Fit source model with Ridge; return coefficients."""
    return Ridge(alpha=lambda_S).fit(X, y).coef_


def pre_trained_lasso(X: np.ndarray, y: np.ndarray, lambda_S: float) -> np.ndarray:
    """Fit source model with Lasso; return coefficients."""
    return Lasso(alpha=lambda_S, max_iter=10000).fit(X, y).coef_


# ── Target adaptation — scipy.minimize formulations ───────────────────────────

def _ridge_transfer_objective(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    theta_S: np.ndarray,
    lambda_T: float,
) -> float:
    residuals = X.dot(theta) - y
    penalty = lambda_T * np.sum((theta - theta_S) ** 2)
    return np.sum(residuals ** 2) / len(y) + penalty


def _ridge_transfer_gradient(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    theta_S: np.ndarray,
    lambda_T: float,
) -> np.ndarray:
    """Analytical gradient of the Ridge transfer objective."""
    return 2 * (X.T @ (X.dot(theta) - y)) / len(y) + 2 * lambda_T * (theta - theta_S)


def _lasso_transfer_objective(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    theta_S: np.ndarray,
    lambda_T: float,
) -> float:
    residuals = X.dot(theta) - y
    penalty = lambda_T * np.sum(np.abs(theta - theta_S))
    return np.sum(residuals ** 2) / len(y) + penalty


def ridge_transfer_coef(
    data: dict, theta_S: np.ndarray, lambda_T: float,
) -> np.ndarray:
    X = data['x_train'].values
    y = data['y_train'].values
    result = minimize(
        _ridge_transfer_objective,
        theta_S.copy(),          # warm start: theta_T is expected near theta_S
        args=(X, y, theta_S, lambda_T),
        jac=_ridge_transfer_gradient,  # analytical gradient avoids numerical diff
    )
    return result.x


def lasso_transfer_coef(
    data: dict, theta_S: np.ndarray, lambda_T: float,
) -> np.ndarray:
    X = data['x_train'].values
    y = data['y_train'].values
    result = minimize(
        _lasso_transfer_objective,
        theta_S.copy(),          # warm start: theta_T is expected near theta_S
        args=(X, y, theta_S, lambda_T),
    )
    return result.x


# ── Metrics ───────────────────────────────────────────────────────────────────

def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float('nan')
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float('nan')
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'mse':  mean_squared_error(y_true, y_pred),
        'r2':   r2_score(y_true, y_pred),
        'corr': _safe_corr(y_true, y_pred),
    }


# ── Per-company runner (top-level function → picklable for joblib) ────────────

def _run_single(
    name: str,
    target: dict,
    X_src: np.ndarray,
    y_src: np.ndarray,
    lambda_S: float,
    lambda_T: float,
    p: float,
    source_trainer,         # pre_trained_ridge or pre_trained_lasso
    target_adapter,         # ridge_transfer_coef or lasso_transfer_coef
    direct_model_cls,       # Ridge or Lasso
) -> tuple:
    """Direct learning + transfer learning for one target company."""

    # ── Source pre-training ──────────────────────────────────────────────────
    theta_S = source_trainer(X_src, y_src, lambda_S)

    # ── Target adaptation ────────────────────────────────────────────────────
    theta_T = target_adapter(target, theta_S, lambda_T)
    y_tl    = pd.Series(target['x_test'].values @ theta_T, index=target['x_test'].index)

    # ── Direct learning ──────────────────────────────────────────────────────
    dl_model = direct_model_cls(alpha=lambda_T)
    dl_model.fit(target['x_train'], target['y_train'])
    y_dl = pd.Series(dl_model.predict(target['x_test']), index=target['x_test'].index)

    # ── Transfer risk (Wasserstein p-distance) ───────────────────────────────
    y_test_arr = target['y_test'].values
    y_ind      = target['x_test'].values @ theta_S
    risk       = wasserstein_distance(y_ind ** p, y_test_arr ** p) ** (1 / p)

    return name, {
        'dl': _metrics(y_test_arr, y_dl),
        'tl': {**_metrics(y_test_arr, y_tl), 'transfer_risk': risk},
        'direct_ci': {
            'y_test':        target['y_test'],
            'y_pred_direct': y_dl,
            'coef':          dl_model.coef_,
        },
        'transfer_ci': {
            'y_test':          target['y_test'],
            'y_pred_transfer': y_tl,
            'theta_S':         theta_S,
            'theta_T':         theta_T,
        },
    }


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiment(
    pp_data:          dict,
    start_dates:      pd.DataFrame,
    lambda_S:         float,
    lambda_T:         float,
    p:                float,
    source_trainer,
    target_adapter,
    direct_model_cls,
    target_start_index: int = 9,
    source_cutoff_fn=None,
) -> dict:
    """
    Run direct + transfer learning for all target companies (index >= 9) in
    parallel.  Source companies for each target are those whose listing date
    is strictly earlier.

    Results are merged back into pp_data in-place and returned.
    """
    targets = list(start_dates.index[target_start_index:])

    if source_cutoff_fn is None:
        def source_cutoff_fn(frame: pd.DataFrame, name: str, absolute_idx: int):
            return frame.loc[name, 'Start date']

    # Build source sub-dicts before parallel dispatch (avoids closure issues).
    # Only pass x_train / y_train to keep pickle size small.
    source_dicts = {
        name: {
            s: {
                'x_train': pp_data[s]['x_train'],
                'y_train': pp_data[s]['y_train'],
            }
            for s in start_dates.index
            if start_dates.loc[s, 'Start date'] < source_cutoff_fn(
                start_dates,
                name,
                start_dates.index.get_loc(name),
            )
        }
        for name in targets
    }

    source_arrays = {
        name: (
            np.vstack([source_dicts[name][src]['x_train'].values for src in source_dicts[name]]),
            np.hstack([source_dicts[name][src]['y_train'].values for src in source_dicts[name]]),
        )
        for name in targets
    }

    results = Parallel(n_jobs=-1)(
        delayed(_run_single)(
            name, pp_data[name], source_arrays[name][0], source_arrays[name][1],
            lambda_S, lambda_T, p,
            source_trainer, target_adapter, direct_model_cls,
        )
        for name in targets
    )

    for name, updates in results:
        pp_data[name].update(updates)

    return pp_data
