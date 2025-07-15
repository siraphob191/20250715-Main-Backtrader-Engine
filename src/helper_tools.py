"""Deprecated module maintained for backward compatibility."""
from utils.data_utils import prepare_benchmark_dataframe
from utils.trade_utils import (
    PortfolioAnalyzer,
    TransactionTracker,
    update_allocation,
    report_target_allocation,
)

__all__ = [
    'prepare_benchmark_dataframe',
    'PortfolioAnalyzer',
    'TransactionTracker',
    'update_allocation',
    'report_target_allocation',
]
