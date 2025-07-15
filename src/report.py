"""Deprecated module maintained for backward compatibility."""
from utils.reporting import (
    annualized_return,
    max_drawdown,
    calculate_sharpe_ratio,
    summarize_portfolio,
    summarize_transactions,
    summarize_benchmark,
)

__all__ = [
    'annualized_return',
    'max_drawdown',
    'calculate_sharpe_ratio',
    'summarize_portfolio',
    'summarize_transactions',
    'summarize_benchmark',
]
