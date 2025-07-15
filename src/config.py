"""Configuration for data paths and default parameters used by the
backtesting scripts."""

import datetime as dt
from pathlib import Path

# Default date range for the backtest
BACKTEST_START_DATE = dt.datetime(1999, 3, 1)
BACKTEST_END_DATE = dt.datetime(2021, 12, 31)

# Default directories mirror the locations used in the original Jupyter
# notebook and legacy script. Update these paths if your data lives
# somewhere else.
etf_data_path = (
    '/Users/siraphobpongkritsagorn/Documents/3 Resources/Historical Data/'
    'Sector ETFs Data'
)
stock_data_path = (
    '/Users/siraphobpongkritsagorn/Documents/3 Resources/Historical Data/'
    'Stocks Data/Raw Data YF format (handled 0 adj close)' # Sample for Testing'
)
benchmark_data_path = (
    '/Users/siraphobpongkritsagorn/Documents/3 Resources/Historical Data/'
    'Benchmark Data'
)

# Additional CSV libraries used by ``run_backtest`` and ``strategy``
sector_library_path = (
    '/Users/siraphobpongkritsagorn/Documents/3 Resources/Historical Data/'
    'Sector Library.csv'
)
sp500_constituents_path = (
    '/Users/siraphobpongkritsagorn/Documents/3 Resources/Historical Data/'
    '20220402 S&P 500 Constituents Symbols.csv'
)
stock_list_path = (
    '/Users/siraphobpongkritsagorn/Documents/3 Resources/Historical Data/'
    'List of Tickers Updated with Sectors.csv'
)

# Broker parameters
# ``run_backtest`` reads these when configuring the Backtrader engine.  They
# can be adjusted in tests via monkeypatching.
INITIAL_CASH = 100000
COMMISSION_PCT = 0.0
SLIPPAGE_PCT = 0.00015

# Benchmark ticker symbol used throughout the project
BENCHMARK_SYMBOL = '^SP500TR'

# Set OUTPUT_DIR to the outputs folder at the root of the repo
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'

# Output CSV filenames used by reporting utilities
PORTFOLIO_CSV = OUTPUT_DIR / 'portfolio.csv'
BENCHMARK_CSV = OUTPUT_DIR / 'benchmark.csv'
TRANSACTIONS_CSV = OUTPUT_DIR / 'transactions.csv'
