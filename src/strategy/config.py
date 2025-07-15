class StrategyConfig(object):
    """Default parameters for the trading strategy."""

    def __init__(
        self,
        rebalance_freq=60,
        init_rebalance_count=58,
        sharpe_period_short=63,
        sharpe_period_medium=126,
        sharpe_period_long=252,
        max_no_of_sectors=4,
        max_sector_weight=1,
        qualify_pct=0.1,
        max_stock_weight=1,
        min_no_of_stocks=3,
        asset_short_sma_period=1,
        asset_long_sma_period=200,
    ):
        # Rebalance parameters
        self.rebalance_freq = rebalance_freq
        self.init_rebalance_count = init_rebalance_count

        # Momentum Score Parameters
        self.sharpe_period_short = sharpe_period_short
        self.sharpe_period_medium = sharpe_period_medium
        self.sharpe_period_long = sharpe_period_long

        # Asset Allocation
        self.max_no_of_sectors = max_no_of_sectors
        self.max_sector_weight = max_sector_weight
        self.qualify_pct = qualify_pct
        self.max_stock_weight = max_stock_weight
        self.min_no_of_stocks = min_no_of_stocks
        self.asset_short_sma_period = asset_short_sma_period
        self.asset_long_sma_period = asset_long_sma_period


# Instance used by strategy classes
DEFAULT_CONFIG = StrategyConfig()

# Tuple of parameter name/value pairs for Backtrader strategy defaults.
DEFAULT_PARAMS = tuple(DEFAULT_CONFIG.__dict__.items())

# Directory containing benchmark CSV files, including ``config.BENCHMARK_SYMBOL.csv``.
MARKET_DATA_PATH = (
    '/Users/siraphobpongkritsagorn/Documents/3 Resources/Historical Data/'
    'Benchmark Data'
)

# Ticker symbol used for the market benchmark
import config
BENCHMARK_SYMBOL = config.BENCHMARK_SYMBOL
