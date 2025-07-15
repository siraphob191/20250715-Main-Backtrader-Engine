"""Strategy package bundling core logic and utilities."""

from .core import SVDMomentumStrategy
from .trend_filter import market_trend_filter

__all__ = [
    'SVDMomentumStrategy',
    'market_trend_filter',
]
