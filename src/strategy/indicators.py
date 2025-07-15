import backtrader as bt


def is_stock_inactive_within_period(stock_data, start_date, end_date, consecutive_days=3):
    """Return True if a stock's price didn't change for ``consecutive_days`` within a period."""
    period_data = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]
    return period_data['Adj Close'].rolling(window=consecutive_days).apply(lambda x: len(set(x)) == 1, raw=True).any()


def calculate_sharpe_ratio(daily_returns, period):
    """Return the rolling Sharpe ratio for ``daily_returns``."""
    rolling_mean = bt.indicators.SMA(daily_returns, period=period)
    rolling_std_dev = bt.indicators.StandardDeviation(daily_returns, period=period)
    epsilon = 1e-8
    return rolling_mean / (rolling_std_dev + epsilon)


def calculate_avg_sharpe_ratio(data, short_period, medium_period, long_period):
    """Return short, medium, long and average Sharpe ratios for ``data``."""
    daily_returns = data.close / data.close(-1) - 1
    short_sr = calculate_sharpe_ratio(daily_returns, short_period)
    medium_sr = calculate_sharpe_ratio(daily_returns, medium_period)
    long_sr = calculate_sharpe_ratio(daily_returns, long_period)
    average_sr = (short_sr + medium_sr + long_sr) / 3
    return {
        'short': short_sr,
        'medium': medium_sr,
        'long': long_sr,
        'average': average_sr,
    }
