import backtrader as bt  # Core backtesting framework
import numpy as np       # Numerical operations
import pandas as pd      # Data handling
import datetime as dt    # Date utilities
from .. import config     # Project-specific configuration

from .market_signal import get_market_signals          # Market signal generator (not used here)
from ..utils.trade_utils import update_allocation       # Update current_alloc after trades

# --------------------------------------------------------------------
# Global variables populated by the backtest runner:
sp500_by_date = {}           # Mapping: date -> set of S&P 500 tickers
etf_tickers = []             # ETF symbols list
stock_tickers = []           # Stock symbols list
benchmark_data = None        # Benchmark history (e.g., SPY)
stock_data_path = ''         # Path to raw stock files

class SVDMomentumStrategy(bt.Strategy):
    """
    Long-only momentum strategy using an SVD-based covariance model.
    Excludes stale-price feeds and enforces long-only, max-weight,
    and turnover constraints on a classic mean-variance solution.
    """
    params = dict(
        rebalance_freq=21,     # Bars between rebalances
        num_factors=50,        # Number of SVD factors used in covariance model
        max_stock_weight=0.20, # Maximum weight for any single stock
        turnover_limit=1.0,    # Maximum total turnover per rebalance
    )

    def __init__(self):
        # 1. Counter for bars since inception
        self.rebalance_count = 0
        # 2. Dictionaries to track current and desired weights
        self.current_alloc = {d: 0 for d in self.datas}
        self.target_alloc = {d: 0 for d in self.datas}
        # 3. Track last rebalance and trade execution dates
        self.rebalance_date = None
        self.transaction_date = None
        # 4. Flag for first valid bar
        self.inception = False

        # 5. Load S&P 500 membership map
        self.sp500_by_date = sp500_by_date
        # 6. Select only data feeds corresponding to stock tickers
        self.stock_datas = [d for d in self.datas if d._name in stock_tickers]

    def prenext(self):
        # Wait until at least one stock feed has 252 bars
        if any(len(d) >= 252 for d in self.stock_datas):
            self.next()

    def nextstart(self):
        # Once all feeds are live, enter the main loop
        self.next()

    def next(self):
        # 1. Require at least one stock with 252 days of history
        if not any(len(d) >= 252 for d in self.stock_datas):
            return

        # 2. Mark the date and increment the counter
        current_date = self.datas[0].datetime.date(0)
        print(f"Date: {current_date}")  # Debug: daily log
        self.inception = True
        self.rebalance_count += 1

        # 3. On scheduled rebalance bars, compute and trade targets
        if self.rebalance_count % self.p.rebalance_freq == 0:
            print(f"Rebalance triggered on {current_date}")
            self.rebalance_date = current_date
            self._compute_target_weights(current_date)
            self._submit_orders(current_date)

        # 4. On T+1 (settlement), update current_alloc based on fills
        if self.rebalance_date and current_date == self.rebalance_date + dt.timedelta(days=1):
            self.transaction_date = current_date
            update_allocation(self)
            print(f"Updated allocation: {self.current_alloc}")

    def _compute_target_weights(self, current_date):
        # 1. Fetch today's S&P 500 constituents
        active = set(self.sp500_by_date.get(current_date, set()))
        # 2. Initial filter: in active set and at least 253 bars of history
        raw_feeds = [d for d in self.stock_datas
                     if d._name in active and len(d) >= 253]
        # 3. Exclude stale-price feeds: three consecutive identical closes
        stock_feeds = []
        for d in raw_feeds:
            # Gather last 253 closing prices (oldest→newest)
            closes = [d.close[-i] for i in range(252, -1, -1)]
            # Compute day-to-day differences
            diffs = np.diff(closes)
            # If any two successive diffs are zero, skip this stale feed
            if any(diffs[i] == 0 and diffs[i+1] == 0 for i in range(len(diffs)-1)):
                continue
            stock_feeds.append(d)
        # 4. Abort if no valid feeds
        if not stock_feeds:
            return

        # 5. Build returns matrix R (252 rows × N stocks)
        returns = []
        names = []
        for d in stock_feeds:
            # Compute past 252 daily returns: (today/prev1)−1
            dr = [(d.close[-i] / d.close[-i-1] - 1) for i in range(1, 253)]
            returns.append(dr[::-1])  # chronological order
            names.append(d)
        R = np.array(returns).T  # shape = (252, N)

        # 6. Compute average return μ for each stock
        raw_mean = R.mean(axis=0)

        # 7. Center R by subtracting μ, then winsorize extremes
        X = R - raw_mean
        for i in range(X.shape[1]):
            low, high = np.quantile(X[:, i], [0.01, 0.99])
            X[:, i] = np.clip(X[:, i], low, high)

        # 8. Perform SVD on X to extract principal factors
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        k = min(self.p.num_factors, len(s))           # number of retained factors
        B = vt.T[:, :k]                              # factor loadings matrix
        Lambda = np.diag((s[:k]**2) / (X.shape[0] - 1))  # factor variances

        # 9. Compute idiosyncratic residuals and variances
        resid = X - X @ B @ B.T
        D = np.diag(resid.var(axis=0, ddof=1))

        # 10. Reconstruct full covariance Σ = BΛBᵀ + D
        Sigma = B @ Lambda @ B.T + D

        # 11. Solve Markowitz: Σ w = μ → raw weights
        try:
            weights = np.linalg.solve(Sigma, raw_mean)
        except np.linalg.LinAlgError:
            # Fallback to equal weights on singular Σ
            weights = np.ones(len(names))

        # 12. Long-only: clip negatives to zero
        weights = np.clip(weights, 0, None)
        # 13. If all zero, revert to equal weights
        if weights.sum() == 0:
            weights = np.ones(len(names))
        # 14. Normalize to sum to 1
        weights /= weights.sum()

        # 15. Cap individual weights at max_stock_weight
        weights = np.minimum(weights, self.p.max_stock_weight)
        # 16. Renormalize after capping
        weights /= weights.sum()

        # 17. Compute total turnover = Σ |w_new − w_old|
        turnover = sum(abs(weights[i] - self.current_alloc.get(names[i], 0))
                       for i in range(len(names)))
        # 18. If turnover exceeds limit, scale changes
        if turnover > self.p.turnover_limit:
            scale = self.p.turnover_limit / turnover
            weights = np.array([
                self.current_alloc.get(names[i], 0)
                + scale * (weights[i] - self.current_alloc.get(names[i], 0))
                for i in range(len(names))
            ])
            # 19. Renormalize after turnover scaling
            weights /= weights.sum()

        # 20. Reset all target allocations from previous rebalance to zero
        for d in self.datas:
            self.target_alloc[d] = 0
        # 21. Assign computed weights to target_alloc for each stock
        for w, d in zip(weights, names):
            self.target_alloc[d] = w

    def _submit_orders(self, current_date):
        # 1. Sell leg: reduce positions where current > target
        self.sell_alloc = {}
        for d in self.stock_datas:
            cw = self.current_alloc[d]         # current weight
            tw = self.target_alloc[d]          # target weight
            if cw > tw:
                self.sell_alloc[d] = tw
                # Market order to trim to target
                self.order_target_percent(d, target=tw,
                                         exectype=bt.Order.Market)
        # 2. Buy leg: add positions where target > current and not sold
        self.buy_alloc = {}
        for d in self.stock_datas:
            if d not in self.sell_alloc:
                tw = self.target_alloc[d]
                if tw > 0:
                    # Market order to build to target
                    self.order_target_percent(d, target=tw,
                                             exectype=bt.Order.Market)
                    self.buy_alloc[d] = tw

# Alias for backward compatibility
TestStrategy = SVDMomentumStrategy
