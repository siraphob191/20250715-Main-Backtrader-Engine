import os
import datetime as dt
import pytest

pandas = pytest.importorskip('pandas')
backtrader = pytest.importorskip('backtrader')

import src.config as config
import src.run_backtest as run_backtest
import src.strategy.trend_filter as market_trend_filter
import src.strategy.config as strategy_config


# Helper to create a price DataFrame

def _price_df(dates):
    """Return a simple OHLCV DataFrame for the given dates."""
    return pandas.DataFrame({
        'Date': dates,
        'Open': [1.0] * len(dates),
        'High': [1.0] * len(dates),
        'Low': [1.0] * len(dates),
        'Close': [1.0] * len(dates),
        'Adj Close': [1.0] * len(dates),
        'Volume': [100] * len(dates)
    })


def _setup_files(tmpdir):
    """Create temporary CSV files expected by ``run_backtest``."""
    dates = pandas.date_range('2020-01-01', periods=260)

    etf_dir = tmpdir.mkdir('etf')
    for ticker in ['XLY', 'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLB', 'XLP', 'XLU', 'XLRE', 'XLC']:
        _price_df(dates).to_csv(etf_dir.join(ticker + '.csv'), index=False)

    stock_dir = tmpdir.mkdir('stocks')
    for ticker in ['AAPL', 'MSFT']:
        _price_df(dates).to_csv(stock_dir.join(ticker + '.csv'), index=False)

    bench_dir = tmpdir.mkdir('benchmark')
    _price_df(dates).to_csv(bench_dir.join('^SP500TR.csv'), index=False)

    pandas.DataFrame({
        'Ticker': ['XLY', 'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLB', 'XLP', 'XLU', 'XLRE', 'XLC'],
        'Sector': ['Sector{}'.format(i) for i in range(11)]
    }).to_csv(tmpdir.join('Sector Library.csv'), index=False)

    pandas.DataFrame({
        '0': pandas.to_datetime(['2020-01-01', '2020-01-02']),
        'AAPL': ['AAPL', 'AAPL'],
        'MSFT': ['MSFT', 'MSFT']
    }).to_csv(tmpdir.join('20220402 S&P 500 Constituents Symbols.csv'), index=False)

    pandas.DataFrame({
        'Custom_Code': ['AAPL', 'MSFT'],
        'GICS Sector': ['Sector0', 'Sector0']
    }).to_csv(tmpdir.join('List of Tickers Updated with Sectors.csv'), index=False)

    return str(etf_dir), str(stock_dir), str(bench_dir)


def test_run_backtest_main(tmp_path, monkeypatch):
    etf_dir, stock_dir, bench_dir = _setup_files(tmp_path)

    monkeypatch.setattr(config, 'etf_data_path', etf_dir)
    monkeypatch.setattr(config, 'stock_data_path', stock_dir)
    monkeypatch.setattr(config, 'benchmark_data_path', bench_dir)
    monkeypatch.setattr(strategy_config, 'MARKET_DATA_PATH', bench_dir)
    monkeypatch.setattr(config, 'sector_library_path', os.path.join(str(tmp_path), 'Sector Library.csv'))
    monkeypatch.setattr(config, 'sp500_constituents_path', os.path.join(str(tmp_path), '20220402 S&P 500 Constituents Symbols.csv'))
    monkeypatch.setattr(config, 'stock_list_path', os.path.join(str(tmp_path), 'List of Tickers Updated with Sectors.csv'))
    monkeypatch.setattr(config, 'BACKTEST_START_DATE', dt.datetime(2020, 1, 1))
    monkeypatch.setattr(config, 'BACKTEST_END_DATE', dt.datetime(2020, 9, 16))
    monkeypatch.setattr(config, 'INITIAL_CASH', 100000)
    monkeypatch.setattr(config, 'COMMISSION_PCT', 0.0)
    monkeypatch.setattr(config, 'SLIPPAGE_PCT', 0.00015)
    monkeypatch.setattr(market_trend_filter.plt, 'show', lambda: None)

    captured = {}
    orig_init = run_backtest.SVDMomentumStrategy.__init__

    def spy_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        captured['mapping'] = self.sp500_by_date
        captured['outside'] = run_backtest.strategy.core.outside_bounds_dates
        captured['trend'] = run_backtest.strategy.core.short_ma_under_long_ma_dates

    monkeypatch.setattr(run_backtest.SVDMomentumStrategy, '__init__', spy_init)

    run_backtest.main()

    assert isinstance(run_backtest.strategy.sp500_by_date, dict)
    assert dt.date(2020, 1, 1) in run_backtest.strategy.sp500_by_date
    assert captured.get('mapping') is run_backtest.strategy.sp500_by_date
    assert isinstance(run_backtest.strategy.core.outside_bounds_dates, list)
    assert isinstance(run_backtest.strategy.core.short_ma_under_long_ma_dates, list)
    assert captured.get('outside') is run_backtest.strategy.core.outside_bounds_dates
    assert (
        captured.get('trend')
        is run_backtest.strategy.core.short_ma_under_long_ma_dates
    )


def test_pyfolio_positions_include_all_tickers(tmp_path, monkeypatch):
    """PyFolio analyzer should track positions for all traded tickers."""

    etf_dir, stock_dir, bench_dir = _setup_files(tmp_path)

    monkeypatch.setattr(config, 'etf_data_path', etf_dir)
    monkeypatch.setattr(config, 'stock_data_path', stock_dir)
    monkeypatch.setattr(config, 'benchmark_data_path', bench_dir)
    monkeypatch.setattr(strategy_config, 'MARKET_DATA_PATH', bench_dir)
    monkeypatch.setattr(
        config,
        'sector_library_path',
        os.path.join(str(tmp_path), 'Sector Library.csv'),
    )
    monkeypatch.setattr(
        config,
        'sp500_constituents_path',
        os.path.join(str(tmp_path), '20220402 S&P 500 Constituents Symbols.csv'),
    )
    monkeypatch.setattr(
        config,
        'stock_list_path',
        os.path.join(str(tmp_path), 'List of Tickers Updated with Sectors.csv'),
    )
    monkeypatch.setattr(config, 'BACKTEST_START_DATE', dt.datetime(2020, 1, 1))
    monkeypatch.setattr(config, 'BACKTEST_END_DATE', dt.datetime(2020, 9, 16))
    monkeypatch.setattr(config, 'INITIAL_CASH', 100000)
    monkeypatch.setattr(config, 'COMMISSION_PCT', 0.0)
    monkeypatch.setattr(config, 'SLIPPAGE_PCT', 0.00015)
    monkeypatch.setattr(market_trend_filter.plt, 'show', lambda: None)

    captured = {}

    def capture_report(df_portfolio, df_benchmark, df_transactions, strat):
        pyf = strat.analyzers.getbyname('pyfolio')
        _, positions, _, _ = pyf.get_pf_items()
        captured['positions'] = positions

    monkeypatch.setattr(run_backtest, 'generate_report', capture_report)

    run_backtest.main()

    positions = captured.get('positions')
    assert positions is not None
    assert 'AAPL' in positions.columns
    assert 'MSFT' in positions.columns


def test_pyfolio_analyzer_tracks_portfolio(tmp_path, monkeypatch):
    """PyFolio should be configured to use portfolio returns."""

    etf_dir, stock_dir, bench_dir = _setup_files(tmp_path)

    monkeypatch.setattr(config, 'etf_data_path', etf_dir)
    monkeypatch.setattr(config, 'stock_data_path', stock_dir)
    monkeypatch.setattr(config, 'benchmark_data_path', bench_dir)
    monkeypatch.setattr(strategy_config, 'MARKET_DATA_PATH', bench_dir)
    monkeypatch.setattr(
        config,
        'sector_library_path',
        os.path.join(str(tmp_path), 'Sector Library.csv'),
    )
    monkeypatch.setattr(
        config,
        'sp500_constituents_path',
        os.path.join(str(tmp_path), '20220402 S&P 500 Constituents Symbols.csv'),
    )
    monkeypatch.setattr(
        config,
        'stock_list_path',
        os.path.join(str(tmp_path), 'List of Tickers Updated with Sectors.csv'),
    )
    monkeypatch.setattr(config, 'BACKTEST_START_DATE', dt.datetime(2020, 1, 1))
    monkeypatch.setattr(config, 'BACKTEST_END_DATE', dt.datetime(2020, 9, 16))
    monkeypatch.setattr(config, 'INITIAL_CASH', 100000)
    monkeypatch.setattr(config, 'COMMISSION_PCT', 0.0)
    monkeypatch.setattr(config, 'SLIPPAGE_PCT', 0.00015)
    monkeypatch.setattr(market_trend_filter.plt, 'show', lambda: None)

    captured = {}

    original_init = run_backtest.btanalyzers.PyFolio.__init__

    def spy_init(self, *args, **kwargs):
        captured['params'] = kwargs
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(run_backtest.btanalyzers.PyFolio, '__init__', spy_init)
    monkeypatch.setattr(run_backtest, 'generate_report', lambda *a, **k: None)

    run_backtest.main()

    params = captured.get('params', {})
    assert params == {}
