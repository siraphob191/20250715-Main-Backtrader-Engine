import os
import backtrader as bt

from .. import config


def load_etf_feeds(tickers, start_date, end_date, path=None):
    """Load ETF data feeds.

    Parameters default to paths defined in :mod:`config` when ``path`` is None.
    """
    if path is None:
        path = config.etf_data_path
    feeds = []
    for ticker in tickers:
        csv_file_path = os.path.join(path, "{}.csv".format(ticker))
        data = bt.feeds.YahooFinanceCSVData(
            dataname=csv_file_path,
            fromdate=start_date,
            todate=end_date,
            name=ticker,
        )
        feeds.append(data)
    return feeds


def load_stock_feeds(start_date, end_date, path=None):
    """Load stock data feeds from all CSV files.

    All stocks are loaded without checking for inactivity.
    """
    if path is None:
        path = config.stock_data_path
    feeds = []
    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            ticker = filename.split('.')[0]
            csv_file_path = os.path.join(path, "{}.csv".format(ticker))
            data = bt.feeds.YahooFinanceCSVData(
                dataname=csv_file_path,
                fromdate=start_date,
                todate=end_date,
                name=ticker,
            )
            feeds.append(data)
    return feeds


def load_benchmark_data(ticker, start_date, end_date, path=None):
    """Load benchmark data feed."""
    if path is None:
        path = config.benchmark_data_path
    csv_file_path = os.path.join(path, "{}.csv".format(ticker))
    return bt.feeds.YahooFinanceCSVData(
        dataname=csv_file_path,
        fromdate=start_date,
        todate=end_date,
        name=ticker,
    )
