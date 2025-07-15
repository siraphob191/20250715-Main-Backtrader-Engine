"""Helpers for preparing data for reporting.

This module contains routines for constructing data frames used by the
reporting utilities. Currently it only provides ``prepare_benchmark_dataframe``
which aligns a benchmark's performance with the portfolio's date range.
"""

import os
import pandas as pd


def prepare_benchmark_dataframe(df_portfolio_data, benchmark_path, ticker):
    """Return benchmark performance aligned with the portfolio dates."""
    benchmark_file_path = os.path.join(benchmark_path, "{}.csv".format(ticker))
    df_benchmark = pd.read_csv(benchmark_file_path)
    df_benchmark['Date'] = pd.to_datetime(df_benchmark['Date'])
    df_benchmark.set_index('Date', inplace=True)
    df_benchmark_filtered = df_benchmark.loc[
        df_benchmark.index.isin(df_portfolio_data.index)
    ].copy()

    if len(df_benchmark_filtered) > 0:
        initial_investment = df_portfolio_data['portfolio_value'].iloc[0]
        first_close = df_benchmark_filtered['Adj Close'].iloc[0]
        df_benchmark_filtered['portfolio_value'] = initial_investment * (
            df_benchmark_filtered['Adj Close'] / first_close
        )
        df_benchmark_data = df_benchmark_filtered[['portfolio_value']].copy()
        df_benchmark_data['accumulated_pl'] = (
            df_benchmark_data['portfolio_value'] - initial_investment
        )
        df_benchmark_data['accumulated_pl_percent'] = (
            df_benchmark_data['accumulated_pl'] / initial_investment
        ) * 100
        df_benchmark_data['daily_pl'] = df_benchmark_data['portfolio_value'].diff()
        df_benchmark_data['daily_pl_percent'] = (
            df_benchmark_data['daily_pl']
            / df_benchmark_data['portfolio_value'].shift(1)
        ) * 100
        df_benchmark_data.fillna(0, inplace=True)
    else:
        df_benchmark_data = pd.DataFrame()

    return df_benchmark_data
