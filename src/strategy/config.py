class StrategyConfig(object):
    """Parameters for the :class:`SVDMomentumStrategy`."""

    def __init__(
        self,
        rebalance_freq=21,
        num_factors=50,
        max_stock_weight=0.05,
        turnover_limit=1.0,
    ):
        # How often the portfolio is rebalanced (trading days)
        self.rebalance_freq = rebalance_freq

        # Number of principal components to keep when estimating the
        # covariance matrix using singular value decomposition
        self.num_factors = num_factors

        # Maximum weight per stock in the portfolio
        self.max_stock_weight = max_stock_weight

        # Limit on portfolio turnover at each rebalance
        self.turnover_limit = turnover_limit


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
from .. import config
BENCHMARK_SYMBOL = config.BENCHMARK_SYMBOL
