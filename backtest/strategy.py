"""
Rolling walk-forward backtest and trading strategy.

Usage (example with hybrid Ridge-Lasso):
    from src.data      import load_data, precompute_signatures, pre_processing_fast
    from src.transfer  import pre_trained_ridge, lasso_transfer_coef, run_experiment
    from backtest.strategy import run_backtest, build_prediction_df, \
                                  add_prediction_intervals, plot_strategy
    from sklearn.linear_model import Ridge

    sig_cache = precompute_signatures(data_frames, start_dates, lags=[2], depths=[2])

    backtest = run_backtest(
        start_dates, data_frames, sig_cache,
        date_range   = pd.date_range('2024-01-01', '2024-11-30', freq='MS'),
        lag=2, depth=2,
        lambda_S=0.0001, lambda_T=0.0005, p=2,
        source_trainer   = pre_trained_ridge,
        target_adapter   = lasso_transfer_coef,
        direct_model_cls = Ridge,
    )

    for stock in ['GM', 'STLA', '9863.HK', '601127.SS']:
        graph = build_prediction_df(backtest, stock, date_range)
        graph = add_prediction_intervals(graph, date_range)
        plot_strategy(stock, data_frames, graph, date_range)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample

from src.data     import pre_processing_fast
from src.transfer import run_experiment


# ── Walk-forward backtest ─────────────────────────────────────────────────────

def run_backtest(
    start_dates:      pd.DataFrame,
    data_frames:      dict,
    sig_cache:        dict,
    date_range:       pd.DatetimeIndex,
    lag:              int,
    depth:            int,
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
    For each split date in date_range, re-preprocess (fast, via cache) and
    re-fit all models.  Returns backtest[date] = pp_data.
    """
    backtest = {}
    for date in date_range:
        pp = pre_processing_fast(
            start_dates, data_frames, sig_cache, lag, depth, date,
        )
        run_experiment(
            pp, start_dates, lambda_S, lambda_T, p,
            source_trainer, target_adapter, direct_model_cls,
            target_start_index=target_start_index,
            source_cutoff_fn=source_cutoff_fn,
        )
        backtest[date] = pp
    return backtest


# ── Build rolling prediction dataframe ───────────────────────────────────────

def build_prediction_df(
    backtest:   dict,
    stock:      str,
    date_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Stitch together one-month-ahead predictions across all backtest dates
    into a single DataFrame with columns y_test, y_pred_direct, y_pred_transfer.
    """
    dff = None
    for date in date_range:
        data  = backtest[date][stock]
        ci    = {**data['direct_ci'], **data['transfer_ci']}
        chunk = pd.DataFrame({
            'y_test':          ci['y_test'],
            'y_pred_direct':   ci['y_pred_direct'],
            'y_pred_transfer': ci['y_pred_transfer'],
        })
        if dff is None:
            dff = chunk
        else:
            dff.update(chunk)
    dff['residuals'] = dff['y_pred_transfer'] - dff['y_test']
    return dff


# ── Bootstrap prediction intervals ───────────────────────────────────────────

def add_prediction_intervals(
    df:           pd.DataFrame,
    date_range:   pd.DatetimeIndex,
    n_bootstrap:  int = 1000,
    alpha:        float = 0.05,
) -> pd.DataFrame:
    """Add lower_bound, upper_bound, and y_pred_mean columns via bootstrap."""
    for d in range(len(date_range) - 1):
        window = df.loc[date_range[d]: date_range[d + 1], 'y_pred_transfer']
        bs = np.array([
            resample(window, n_samples=len(window)).values
            for _ in range(n_bootstrap)
        ])
        df.loc[date_range[d]: date_range[d + 1], 'lower_bound'] = \
            np.percentile(bs, 100 * alpha / 2,       axis=0)
        df.loc[date_range[d]: date_range[d + 1], 'upper_bound'] = \
            np.percentile(bs, 100 * (1 - alpha / 2), axis=0)
        df.loc[date_range[d]: date_range[d + 1], 'y_pred_mean'] = \
            np.mean(bs, axis=0)
    return df.bfill()


# ── Trading signal ────────────────────────────────────────────────────────────

def _adjust_signal(signal: np.ndarray) -> np.ndarray:
    """
    Suppress consecutive -1 (sell) signals: only the first sell in a streak
    is kept; subsequent -1s become 0 until the next buy (+1).
    """
    out = signal.copy()
    in_sell = False
    for i in range(1, len(signal)):
        if signal[i] == -1:
            if in_sell:
                out[i] = 0
            else:
                in_sell = True
        elif signal[i] == 1:
            in_sell = False
    return out


def compute_strategy_metrics(
    graph: pd.DataFrame,
    data_frames: dict,
    stock: str,
    date_range: pd.DatetimeIndex,
    trading_days: int = 252,
) -> dict:
    """Compute cumulative return, annualized return, Sharpe, and max drawdown."""
    y_raw = data_frames[stock]['Close'].diff()[1:]
    std   = y_raw.std()
    mean  = y_raw.mean()

    initial_prices = pd.Series(
        [data_frames[stock]['Close'].loc[:d].iloc[-1] for d in date_range],
        index=date_range,
    )

    line3 = [_reprice(graph['y_pred_transfer'].loc[d:], initial_prices.iloc[i], std, mean)
             for i, d in enumerate(date_range)]
    for i in range(1, len(line3)):
        line3[0].update(line3[i])

    line4 = [_reprice(graph['y_pred_mean'].loc[d:], initial_prices.iloc[i], std, mean)
             for i, d in enumerate(date_range)]
    for i in range(1, len(line4)):
        line4[0].update(line4[i])

    price_actual = _reprice(graph['y_test'], initial_prices.iloc[0], std, mean)
    signal = np.where(line3[0] < line4[0], 1, -1)
    adjusted_signal = _adjust_signal(signal)
    returns = np.log(price_actual / price_actual.shift(1)).dropna()
    strategy_ret = adjusted_signal[:-1] * returns.iloc[:len(adjusted_signal) - 1]
    strategy_ret = pd.Series(strategy_ret, index=returns.index[:len(strategy_ret)])
    cum_ret = np.exp(strategy_ret.cumsum())

    total_return = float(cum_ret.iloc[-1] - 1) if len(cum_ret) else float('nan')
    annualized_return = (
        float(cum_ret.iloc[-1] ** (trading_days / len(cum_ret)) - 1)
        if len(cum_ret) else float('nan')
    )
    sharpe = (
        float(strategy_ret.mean() / strategy_ret.std() * np.sqrt(trading_days))
        if len(strategy_ret) and strategy_ret.std() != 0 else float('nan')
    )
    max_drawdown = (
        float((cum_ret / cum_ret.cummax() - 1).min())
        if len(cum_ret) else float('nan')
    )

    return {
        'cumulative_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'num_days': int(len(strategy_ret)),
    }


def compute_equity_curve(
    graph: pd.DataFrame,
    data_frames: dict,
    stock: str,
    date_range: pd.DatetimeIndex,
) -> pd.Series:
    """Build the cumulative equity curve for the trading strategy."""
    y_raw = data_frames[stock]['Close'].diff()[1:]
    std   = y_raw.std()
    mean  = y_raw.mean()

    initial_prices = pd.Series(
        [data_frames[stock]['Close'].loc[:d].iloc[-1] for d in date_range],
        index=date_range,
    )

    line3 = [_reprice(graph['y_pred_transfer'].loc[d:], initial_prices.iloc[i], std, mean)
             for i, d in enumerate(date_range)]
    for i in range(1, len(line3)):
        line3[0].update(line3[i])

    line4 = [_reprice(graph['y_pred_mean'].loc[d:], initial_prices.iloc[i], std, mean)
             for i, d in enumerate(date_range)]
    for i in range(1, len(line4)):
        line4[0].update(line4[i])

    price_actual = _reprice(graph['y_test'], initial_prices.iloc[0], std, mean)
    signal = np.where(line3[0] < line4[0], 1, -1)
    adjusted_signal = _adjust_signal(signal)
    returns = np.log(price_actual / price_actual.shift(1)).dropna()
    strategy_ret = adjusted_signal[:-1] * returns.iloc[:len(adjusted_signal) - 1]
    strategy_ret = pd.Series(strategy_ret, index=returns.index[:len(strategy_ret)])
    return np.exp(strategy_ret.cumsum())


def _reprice(
    returns:       pd.Series,
    initial_price: float,
    std:           float,
    mean:          float,
) -> pd.Series:
    log_returns = returns * std + mean
    return np.exp(initial_price + log_returns.cumsum())


# ── Strategy plot ─────────────────────────────────────────────────────────────

def plot_strategy(
    stock:        str,
    data_frames:  dict,
    graph:        pd.DataFrame,
    date_range:   pd.DatetimeIndex,
    split_date=   pd.Timestamp('2024-01-01'),
) -> None:
    """
    Two-panel plot:
      Top:    actual price vs. transfer-learning prediction + bootstrap mean
      Bottom: cumulative strategy returns with Sharpe and max drawdown
    """
    y_raw = data_frames[stock]['Close'].diff()[1:]
    std   = y_raw.std()
    mean  = y_raw.mean()

    initial_prices = pd.Series(
        [data_frames[stock]['Close'].loc[:d].iloc[-1] for d in date_range],
        index=date_range,
    )

    # Stitch monthly repriced predictions
    line3 = [_reprice(graph['y_pred_transfer'].loc[d:], initial_prices.iloc[i], std, mean)
             for i, d in enumerate(date_range)]
    for i in range(1, len(line3)):
        line3[0].update(line3[i])

    line4 = [_reprice(graph['y_pred_mean'].loc[d:], initial_prices.iloc[i], std, mean)
             for i, d in enumerate(date_range)]
    for i in range(1, len(line4)):
        line4[0].update(line4[i])

    price_actual = _reprice(graph['y_test'], initial_prices.iloc[0], std, mean)

    # Trading signals
    signal          = np.where(line3[0] < line4[0], 1, -1)
    adjusted_signal = _adjust_signal(signal)
    returns         = np.log(price_actual / price_actual.shift(1))
    strategy_ret    = adjusted_signal[:-1] * returns.iloc[1:]
    cum_ret         = np.exp(strategy_ret.cumsum())

    sharpe       = strategy_ret.mean() / strategy_ret.std() * np.sqrt(252)
    max_drawdown = (cum_ret - cum_ret.cummax()).min()

    # Plot
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10),
        gridspec_kw={'height_ratios': [3, 1]},
    )

    ax1.plot(graph.index, price_actual,  label='Actual Price',       color='blue')
    ax1.plot(graph.index, line3[0],      label='Transfer Learning',  color='red')
    ax1.plot(graph.index, line4[0],      label='Bootstrap Mean',     color='green',  linestyle='--')
    ax1.set_title(f'Price Prediction — {stock}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()

    label = (
        f'Cumulative Return: {cum_ret.iloc[-1]:.2%} | '
        f'Sharpe: {sharpe:.2f} | '
        f'Max Drawdown: {max_drawdown:.2%}'
    )
    ax2.plot(cum_ret, label=label, color='purple')
    ax2.set_title('Strategy Cumulative Returns')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Return')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'backtest_{stock}.png', dpi=150)
    plt.show()
