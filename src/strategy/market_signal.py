import os
import pandas as pd

from . import config as strategy_config
from .trend_filter import market_trend_filter


def _load_spx_data():
    """Load the S&P500 Total Return index from ``MARKET_DATA_PATH``."""
    file_path = os.path.join(
        strategy_config.MARKET_DATA_PATH,
        f"{strategy_config.BENCHMARK_SYMBOL}.csv",
    )
    return pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')


def get_market_signals(rolling_window=252, sd_range=5,
                       short_ma_window=1, long_ma_window=200):
    """Return market signal dates used by the strategy.

    Parameters
    ----------
    rolling_window : int, optional
        Window size to compute volatility bands.
    sd_range : int, optional
        Standard deviation multiplier for the bands.
    short_ma_window : int, optional
        Short moving average window for the trend filter.
    long_ma_window : int, optional
        Long moving average window for the trend filter.

    Returns
    -------
    tuple[list[pd.Timestamp], list[pd.Timestamp]]
        A tuple ``(outside_bounds_dates, short_ma_under_long_ma_dates)``.
    """
    spx_df = _load_spx_data()

    # Extreme daily move signal
    spx_df['Daily_Return'] = spx_df['Adj Close'].pct_change()
    spx_df['Rolling_Mean'] = spx_df['Daily_Return'].rolling(window=rolling_window).mean()
    spx_df['Rolling_Std'] = spx_df['Daily_Return'].rolling(window=rolling_window).std()
    spx_df['Upper_Bound'] = spx_df['Rolling_Mean'] + (sd_range * spx_df['Rolling_Std'])
    spx_df['Lower_Bound'] = spx_df['Rolling_Mean'] - (sd_range * spx_df['Rolling_Std'])
    spx_df['Outside_Bounds'] = (
        (spx_df['Daily_Return'] > spx_df['Upper_Bound']) |
        (spx_df['Daily_Return'] < spx_df['Lower_Bound'])
    )
    outside_bounds_dates = spx_df[spx_df['Outside_Bounds']].index.tolist()

    # Trend filter signal
    short_ma_under_long_ma_dates = list(
        market_trend_filter(
            spx_df,
            short_ma_window=short_ma_window,
            long_ma_window=long_ma_window,
            show_plot=False
        )
    )

    return outside_bounds_dates, short_ma_under_long_ma_dates
