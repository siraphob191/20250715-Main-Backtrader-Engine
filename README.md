# Momentum 4.6 Refactor

A minimal backtesting framework that implements a momentum strategy for S&P 500 stocks and sector ETFs. The original research notebook lives in `reference/`, while the refactored Python modules reside under `src/`.

## Features

- **src/data/** – loaders, S&P 500 universe and sector mapping helpers
- **src/strategy/** – core trading logic, indicators and filters
- **src/run_backtest.py** – script that executes a single backtest run
- **src/utils/** – utility modules used by `run_backtest`
  - `data_utils.py` prepares benchmark data for comparison
  - `trade_utils.py` reports allocations and defines Backtrader analyzers
  - `reporting.py` summarizes portfolio, transaction and benchmark results
  - `report_io.py` saves CSV files and plots performance
  - The first feed added to `cerebro` is treated as the strategy data feed and
    the S&P 500 CSV provides the benchmark feed for PyFolio comparisons.

## Prerequisites & Installation

 - Python 3.5 or higher

Create and activate a virtual environment:

```bash
python -m venv .env
source .env/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Optionally install the project in editable mode (a minimal `setup.py` is provided) so `src` can be imported as a package:

```bash
pip install -e .
```

## Directory Structure

```
/src
  ├─ data/        # loaders, SP500 universe, sector mapping
  ├─ strategy/    # core logic, signals, filters, indicators
  ├─ utils/       # data preparation, analyzers and reporting modules
  └─ run_backtest.py
/tests            # unit tests
/reference        # legacy notebook & scripts (to archive later)
```

## Usage Examples

In a Jupyter Notebook:

First install the project in editable mode so the `src` package is available:

```bash
pip install -e .
```

Run the backtest from the command line:

```bash
python -m src.run_backtest
```

Or programmatically:

```python
import src.config as config
import src.run_backtest as run_backtest

# override configuration options here
config.BACKTEST_START_DATE = ...
run_backtest.main()
```

Command-line entry points may be added later to run the momentum backtest directly from the shell.

## Running Tests

Install the packages listed in `requirements-test.txt` (which includes `pandas` and `backtrader`) and execute the suite:

```bash
pip install -r requirements-test.txt
pytest tests/
```

All tests should pass (they will be skipped if optional dependencies such as `pandas` or `backtrader` are unavailable).

## Additional Notes

Legacy notebooks remain under `reference/` and are not actively maintained. Badges for build status, coverage and PyPI will be added once CI is set up.

## Troubleshooting

### PyFolio: unexpected `gross_lev` kwarg
- **Error:** create_full_tear_sheet() got an unexpected keyword argument 'gross_lev'
- **Cause:** Current PyFolio no longer accepts a `gross_lev` parameter.
- **Fix:** Removed `gross_lev` from the tear-sheet call.
- **Leverage plots:** If you need gross leverage, plot it separately via:
  ```python
  from pyfolio import plotting
  plotting.plot_leverage(returns, positions, transactions)
  ```

For a full list of known issues and fixes, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
