"""
Result aggregation and Excel export utilities.
"""

import numpy as np
import pandas as pd


def build_swapped_dict(outputs: dict) -> dict:
    """
    Transpose outputs from {lag_depth_key: {company: data}}
    to             {company: {lag_depth_key: data}}.
    """
    swapped: dict = {}
    for outer_key, inner_dict in outputs.items():
        for company, value in inner_dict.items():
            swapped.setdefault(company, {})[outer_key] = value
    return swapped


def save_performance_excel(outputs: dict, path: str) -> None:
    """
    Save per-company performance tables (MSE / R² / Corr for direct and
    transfer learning, plus transfer risk) to an Excel file.

    Each company gets its own sheet with rows = (lag, depth) combinations.
    """
    swapped = build_swapped_dict(outputs)

    pairs = []
    for key in outputs.keys():
        try:
            lag = int(key.split(',')[0].split('=')[1])
            depth = int(key.split(',')[1].split('=')[1])
        except (IndexError, ValueError):
            continue
        pairs.append((lag, depth))

    lags = sorted({lag for lag, _ in pairs})
    depths = sorted({depth for _, depth in pairs})

    index   = pd.MultiIndex.from_product([lags, depths], names=['Lag', 'Depth'])
    columns = pd.MultiIndex.from_tuples([
        ('Direct',   'MSE'),
        ('Direct',   'R²'),
        ('Direct',   'Corr'),
        ('Transfer', 'MSE'),
        ('Transfer', 'R²'),
        ('Transfer', 'Corr'),
        ('Transfer', 'Transfer Risk'),
    ])

    with pd.ExcelWriter(path) as writer:
        for company, ld_dict in swapped.items():
            df = pd.DataFrame(np.nan, index=index, columns=columns)
            for lm_key, data in ld_dict.items():
                if 'dl' not in data or 'tl' not in data:
                    continue
                # Parse lag and depth from key like 'L=2,M=3'
                try:
                    lag   = int(lm_key.split(',')[0].split('=')[1])
                    depth = int(lm_key.split(',')[1].split('=')[1])
                except (IndexError, ValueError):
                    continue
                row = (
                    list(data['dl'].values()) +
                    list(data['tl'].values())
                )
                df.loc[(lag, depth), :] = row
            df.sort_index(inplace=True)
            df.to_excel(writer, sheet_name=str(company)[:31])


def save_coefficients_excel(
    outputs: dict,
    path: str,
    target_stocks: list,
    max_coef_len: int = 2000,
) -> None:
    """
    Save θ_direct, θ_S, θ_T coefficient vectors for selected stocks.

    Each stock gets its own sheet; columns are a MultiIndex
    (lag_depth_key, theta_type).
    """
    swapped = build_swapped_dict(outputs)

    cc = [
        (key, kind)
        for key  in outputs.keys()
        for kind in ('theta_d', 'theta_S', 'theta_T')
    ]
    columns = pd.MultiIndex.from_tuples(cc)

    with pd.ExcelWriter(path) as writer:
        for company in target_stocks:
            if company not in swapped:
                continue
            df = pd.DataFrame(np.nan, index=range(max_coef_len), columns=columns)
            for lm_key, data in swapped[company].items():
                if 'direct_ci' not in data or 'transfer_ci' not in data:
                    continue
                d = data['direct_ci']['coef']
                s = data['transfer_ci']['theta_S']
                t = data['transfer_ci']['theta_T']
                df.loc[:len(d) - 1, (lm_key, 'theta_d')] = d
                df.loc[:len(s) - 1, (lm_key, 'theta_S')] = s
                df.loc[:len(t) - 1, (lm_key, 'theta_T')] = t
            df.to_excel(writer, sheet_name=str(company)[:31])
