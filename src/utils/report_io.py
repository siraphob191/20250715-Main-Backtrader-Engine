"""Persist portfolio results and optionally create plots.

``generate_report`` writes portfolio, benchmark and transaction data to CSV
files and, when optional dependencies are available, produces visualizations of
the backtest using Matplotlib and PyFolio.
"""

try:
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    plt = None

try:  # pragma: no cover - optional dependency
    import pyfolio as pf  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    pf = None

import pandas as pd

from .. import config


def _visualize_results(df_portfolio_data, df_benchmark_data, strat):
    """Plot portfolio vs benchmark and display the PyFolio tear sheet."""

    if plt is not None:
        plt.figure()
        df_portfolio_data['portfolio_value'].plot(
            label='Strategy', color='grey'
        )
        if not df_benchmark_data.empty:
            df_benchmark_data['portfolio_value'].plot(
                label='Benchmark', color='green'
            )
        plt.legend()
        plt.title('Portfolio vs Benchmark')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.tight_layout()
        plt.show()

    if pf is not None:
        try:
            pyfolio_analyzer = strat.analyzers.getbyname('pyfolio')
        except Exception:
            pyfolio_analyzer = None

        if pyfolio_analyzer is not None:
            returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()
            bench_rets = (
                df_benchmark_data['portfolio_value'].pct_change().fillna(0)
            )
            # pyfolio expects matching timezone awareness for all series
            returns.index = pd.DatetimeIndex(returns.index).tz_localize(None)
            bench_rets.index = pd.DatetimeIndex(bench_rets.index).tz_localize(None)

            # Rename the series for clarity in the tear sheet
            returns = returns.rename("Strategy")
            bench_rets = bench_rets.rename("Benchmark")

            # Display which series is which before plotting
            print("returns.name =", returns.name)
            print("benchmark_rets.name =", bench_rets.name)

            pf.create_full_tear_sheet(
                returns.rename("Strategy"),
                positions=positions,
                transactions=transactions,
                benchmark_rets=bench_rets.rename("Benchmark"),
                estimate_intraday=False, # Supporting non-intra-day strategies
            )
            if plt is not None:
                plt.show()


def generate_report(df_portfolio_data, df_benchmark_data, df_transaction_data, strat):
    """Persist results to CSV and generate visual reports."""

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_portfolio_data.to_csv(config.PORTFOLIO_CSV)
    df_benchmark_data.to_csv(config.BENCHMARK_CSV)
    df_transaction_data.to_csv(config.TRANSACTIONS_CSV)

    _visualize_results(df_portfolio_data, df_benchmark_data, strat)
