"""Backtrader entry point used by the tests.

This script loads CSV data for ETFs, stocks and benchmark, sets up the
``SVDMomentumStrategy`` with the required libraries and then runs a single
backtest session. Results are printed to the console.
"""

import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd
from IPython.display import display
import datetime as dt
import os

from . import config

from .data.loader import load_etf_feeds, load_stock_feeds, load_benchmark_data
from .data.sector import load_sector_library, load_stock_list
from .data.sp500 import load_sp500_by_date
from . import strategy
from .strategy import SVDMomentumStrategy
from .utils.trade_utils import PortfolioAnalyzer, TransactionTracker
from .utils.data_utils import prepare_benchmark_dataframe
from .utils.reporting import (
    summarize_portfolio,
    summarize_transactions,
    summarize_benchmark,
)
from .utils.report_io import generate_report


def main():
    """Run a single backtest using the :class:`SVDMomentumStrategy` class.

    The function loads CSV price data, prepares sector and stock mappings and
    configures the Backtrader engine.  After running the simulation it prints
    a summary of portfolio performance, executed transactions and a benchmark
    comparison.
    """
    # The list of sector ETF tickers is derived from the available CSV files in
    # the configured directory.
    etf_data_path = config.etf_data_path
    etf_tickers = [
        os.path.splitext(f)[0]
        for f in os.listdir(etf_data_path)
        if f.endswith('.csv')
    ]
    stock_data_path = config.stock_data_path
    benchmark_data_path = config.benchmark_data_path

    start_date = config.BACKTEST_START_DATE
    end_date = config.BACKTEST_END_DATE

    # Load feeds
    data_feeds = []
    data_feeds.extend(load_etf_feeds(etf_tickers, start_date, end_date))
    data_feeds.extend(load_stock_feeds(start_date, end_date))
    benchmark_data = load_benchmark_data(
        config.BENCHMARK_SYMBOL, start_date, end_date
    )
    data_feeds.append(benchmark_data)

    # Sector and stock libraries
    sector_library_path = config.sector_library_path
    sp500_constituents_path = config.sp500_constituents_path
    stock_list_path = config.stock_list_path

    # Prepare sector library matched with sector ETFs
    sector_library, sector_mapping = load_sector_library(sector_library_path)
    # Prepare sector library matched with stocks
    stock_list, stock_sector_map = load_stock_list(stock_list_path)

    etf_to_sector = sector_mapping
    stock_to_sector = stock_sector_map

    print("=== ETF → Sector mapping (sample) ===")
    display(pd.DataFrame(list(etf_to_sector.items()), columns=["Ticker", "Sector"]).head())
    print("=== Stock → Sector mapping (sample) ===")
    display(pd.DataFrame(list(stock_to_sector.items()), columns=["Ticker", "Sector"]).head())

    # Loads the list of dynamic S&P500 constituents
    sp500_constituents = pd.read_csv(sp500_constituents_path, parse_dates=[0])
    sp500_constituents['0'] = sp500_constituents['0'].dt.date
    sp500_by_date = load_sp500_by_date(sp500_constituents_path)

    print("=== S&P 500 constituents (sample) ===")
    display(sp500_constituents.head())
    print("=== Constituents by date (first 5 dates) ===")
    display(pd.DataFrame.from_dict(sp500_by_date, orient="index").head())

    # Collecting a list of stock tickers
    stock_tickers = []
    for filename in os.listdir(stock_data_path):
        if filename.endswith('.csv'):
            ticker = os.path.splitext(filename)[0]
            stock_tickers.append(ticker)

    # expose global non-strategy-specific variables to strategy.core module
    strategy.core.sector_library = sector_library
    strategy.core.sector_mapping = sector_mapping
    strategy.core.sp500_constituents = sp500_constituents
    strategy.core.sp500_by_date = sp500_by_date
    # also expose mapping on the package for tests
    strategy.sp500_by_date = sp500_by_date
    strategy.core.stock_list = stock_list
    strategy.core.stock_sector_map = stock_sector_map
    strategy.core.etf_tickers = etf_tickers
    strategy.core.stock_tickers = stock_tickers
    strategy.core.benchmark_data = benchmark_data
    strategy.core.stock_data_path = stock_data_path

    # Broker configuration
    initial_cash = config.INITIAL_CASH
    commission_pct = config.COMMISSION_PCT
    slippage_pct = config.SLIPPAGE_PCT

    cerebro = bt.Cerebro()
    for data in data_feeds:
        cerebro.adddata(data)
    cerebro.addstrategy(SVDMomentumStrategy)
    cerebro.broker.set_cash(initial_cash)
    cerebro.broker.setcommission(commission=commission_pct)
    cerebro.broker.set_slippage_perc(slippage_pct, slip_open=True, slip_limit=True, slip_match=True, slip_out=True)
    cerebro.addanalyzer(PortfolioAnalyzer, _name='portfolio_data')
    cerebro.addanalyzer(TransactionTracker, _name='transaction_data')

    strategy_feed = data_feeds[0]
    benchmark_feed = benchmark_data
    # Attach PyFolio: maps our strategy P&L to returns,
    # and S&P 500 feed to benchmark_rets for tear-sheet plots.
    cerebro.addanalyzer(
        btanalyzers.PyFolio,
        _name='pyfolio'
    )

    results = cerebro.run()

    portfolio_data = results[0].analyzers.getbyname('portfolio_data').get_analysis()
    transaction_data = results[0].analyzers.getbyname('transaction_data').get_analysis()
    pyf_analyzer = results[0].analyzers.getbyname('pyfolio')

    df_portfolio_data = pd.DataFrame(portfolio_data)
    df_portfolio_data.set_index('date', inplace=True)
    df_transaction_data = pd.DataFrame(transaction_data)
    if not df_transaction_data.empty:
        df_transaction_data.set_index('date', inplace=True)

    # Ensure PyFolio positions include all tickers even if not traded
    pos_headers = stock_tickers + ['^SP500TR', 'cash']
    pos_dict = {'Datetime': pos_headers}
    for dt_index in df_portfolio_data.index:
        pos_dict[dt_index] = [0.0] * len(pos_headers)
    pyf_analyzer.rets['positions'] = pos_dict

    import types
    from backtrader.utils.py3 import iteritems

    def get_pf_items_all(self):
        import pandas
        from pandas import DataFrame as DF

        cols = ['index', 'return']
        returns = DF.from_records(iteritems(self.rets['returns']), index=cols[0], columns=cols)
        returns.index = pandas.to_datetime(returns.index).tz_localize('UTC')
        rets = returns['return']

        pss = self.rets['positions']
        ps = [[k] + v for k, v in iteritems(pss)]
        cols = ps.pop(0)
        positions = DF.from_records(ps, index=cols[0], columns=cols)
        positions.index = pandas.to_datetime(positions.index).tz_localize('UTC')

        txss = self.rets['transactions']
        txs = []
        for k, v in iteritems(txss):
            for v2 in v:
                txs.append([k] + v2)
        cols = txs.pop(0)
        transactions = DF.from_records(txs, index=cols[0], columns=cols)
        transactions.index = pandas.to_datetime(transactions.index).tz_localize('UTC')

        cols = ['index', 'gross_lev']
        gross_lev = DF.from_records(iteritems(self.rets['gross_lev']), index=cols[0], columns=cols)
        gross_lev.index = pandas.to_datetime(gross_lev.index).tz_localize('UTC')
        glev = gross_lev['gross_lev']

        return rets, positions, transactions, glev

    pyf_analyzer.get_pf_items = types.MethodType(get_pf_items_all, pyf_analyzer)

    df_benchmark_data = prepare_benchmark_dataframe(
        df_portfolio_data,
        benchmark_data_path,
        config.BENCHMARK_SYMBOL,
    )

    portfolio_summary = summarize_portfolio(df_portfolio_data)
    transaction_summary = summarize_transactions(df_transaction_data)
    benchmark_summary = summarize_benchmark(df_benchmark_data, df_portfolio_data)

    print('Portfolio Performance Summary:')
    print(portfolio_summary)
    print('\nTransaction Summary:')
    print(transaction_summary)
    print('\nBenchmark Performance Summary:')
    print(benchmark_summary)

    # Persist results and generate visual reports
    generate_report(
        df_portfolio_data,
        df_benchmark_data,
        df_transaction_data,
        results[0],
    )


if __name__ == '__main__':
    main()
