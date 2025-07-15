"""Summary statistics for portfolios, transactions and benchmarks.

Functions here compute common performance metrics from a portfolio's value
series, describe executed transactions and compare results to a benchmark.
"""

import numpy as np
import pandas as pd


def annualized_return(df):
    """Return annualized portfolio return from a value time series."""
    total_period = (df.index[-1] - df.index[0]).days / 365.25
    ending_value = df['portfolio_value'].iloc[-1]
    starting_value = df['portfolio_value'].iloc[0]
    return ((ending_value / starting_value) ** (1 / total_period)) - 1


def max_drawdown(df):
    """Return the maximum drawdown of ``portfolio_value``."""
    roll_max = df['portfolio_value'].cummax()
    drawdown = df['portfolio_value'] / roll_max - 1.0
    return drawdown.min()


def calculate_sharpe_ratio(df):
    """Return the daily Sharpe ratio of ``portfolio_value``."""
    returns = df['portfolio_value'].pct_change()
    return returns.mean() / returns.std() * np.sqrt(252)


def summarize_portfolio(df):
    """Return key performance metrics for the portfolio."""
    total_returns = (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0]) - 1
    ann_returns = annualized_return(df)
    max_dd = max_drawdown(df)
    sharpe = calculate_sharpe_ratio(df)
    std_dev = df['portfolio_value'].pct_change().std() * np.sqrt(252)
    return pd.DataFrame({
        'Total Returns': [total_returns],
        'Annualized Returns': [ann_returns],
        'Max Drawdown': [max_dd],
        'Sharpe Ratio': [sharpe],
        'Standard Deviation': [std_dev]
    })


def summarize_transactions(df):
    """Return statistics describing executed trades."""
    total_trades = df.shape[0]
    average_trade_size = df['size'].abs().mean()
    total_commissions_paid = df['commission'].sum()
    profitable_trades = df[df['size'] * df['price'] > 0]
    loss_trades = df[df['size'] * df['price'] < 0]
    num_profitable_trades = profitable_trades.shape[0]
    percent_profitable_trades = (num_profitable_trades / total_trades) * 100
    num_loss_trades = loss_trades.shape[0]
    percent_loss_trades = (num_loss_trades / total_trades) * 100
    return pd.DataFrame({
        'Total Trades': [total_trades],
        'Average Trade Size': [average_trade_size],
        'Total Commissions Paid': [total_commissions_paid],
        'Profitable Trades': [num_profitable_trades],
        'Profitable Trades %': [percent_profitable_trades],
        'Loss Trades': [num_loss_trades],
        'Loss Trades %': [percent_loss_trades]
    })


def summarize_benchmark(df, df_portfolio):
    """Return performance statistics for the benchmark."""
    total_returns = (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0]) - 1
    ann_returns = annualized_return(df)
    max_dd = max_drawdown(df)
    sharpe = calculate_sharpe_ratio(df)
    std_dev = df['portfolio_value'].pct_change().std() * np.sqrt(252)
    return pd.DataFrame({
        'Total Returns': [total_returns],
        'Annualized Returns': [ann_returns],
        'Max Drawdown': [max_dd],
        'Sharpe Ratio': [sharpe],
        'Standard Deviation': [std_dev]
    })
