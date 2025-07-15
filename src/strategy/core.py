import backtrader as bt
import numpy as np
import pandas as pd
import datetime as dt
from .. import config

from .market_signal import get_market_signals
from ..utils.trade_utils import update_allocation

# Global variables populated by ``run_backtest``
sp500_by_date = {}
etf_tickers = []
stock_tickers = []
benchmark_data = None
stock_data_path = ''

# Market signal dates used by the strategy. ``get_market_signals`` will populate
# these when the strategy instance is created.
outside_bounds_dates = []
short_ma_under_long_ma_dates = []


class SVDMomentumStrategy(bt.Strategy):
    """Long-only momentum portfolio using an SVD-based covariance model."""

    params = dict(
        rebalance_freq=1,
        num_factors=50,
        max_stock_weight=0.05,
        turnover_limit=1.0,
    )

    def __init__(self):
        self.rebalance_count = 0
        self.current_alloc = {data: 0 for data in self.datas}
        self.target_alloc = {data: 0 for data in self.datas}
        self.rebalance_date = None
        self.transaction_date = None
        self.inception = False

        # Store loaded mapping of S&P 500 constituents
        self.sp500_by_date = sp500_by_date

        global outside_bounds_dates, short_ma_under_long_ma_dates
        outside_bounds_dates, short_ma_under_long_ma_dates = get_market_signals()
        self.outside_bounds_dates = outside_bounds_dates
        self.short_ma_under_long_ma_dates = short_ma_under_long_ma_dates

        self.stock_datas = [d for d in self.datas if d._name in stock_tickers]

    def prenext(self):
        """Wait until all stocks have enough history before trading."""
        if all(len(d) >= 252 for d in self.stock_datas):
            self.next()

    def nextstart(self):
        """Called once when all datas fulfill length requirements."""
        self.next()

    def next(self):
        # Skip until all stocks have at least 252 days of history
        if any(len(d) < 252 for d in self.stock_datas):
            return

        current_date = self.datas[0].datetime.date(0)
        self.inception = True
        self.rebalance_count += 1

        if self.rebalance_count % self.p.rebalance_freq == 0:
            self.rebalance_date = current_date
            self._compute_target_weights(current_date)
            self._submit_orders(current_date)

        # Day after rebalancing update allocation
        if self.rebalance_date and current_date == self.rebalance_date + dt.timedelta(days=1):
            self.transaction_date = current_date
            update_allocation(self)

    # ------------------------------------------------------------------
    def _compute_target_weights(self, current_date):
        active = set(self.sp500_by_date.get(current_date, set()))
        stock_feeds = [d for d in self.stock_datas if d._name in active and len(d) >= 253]
        if not stock_feeds:
            return

        returns = []
        names = []
        for data in stock_feeds:
            daily = [data.close[-i] / data.close[-i-1] - 1 for i in range(1, 253)]
            returns.append(daily[::-1])
            names.append(data)
        R = np.array(returns).T  # shape (252, N)
        raw_mean = R.mean(axis=0)

        # Demean and winsorize
        X = R - R.mean(axis=0)
        for i in range(X.shape[1]):
            low, high = np.quantile(X[:, i], [0.01, 0.99])
            X[:, i] = np.clip(X[:, i], low, high)

        # SVD factor model
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        k = min(self.p.num_factors, len(s))
        B = vt.T[:, :k]
        Lambda = np.diag((s[:k] ** 2) / (X.shape[0] - 1))
        resid = X - X @ B @ B.T
        D = np.diag(resid.var(axis=0, ddof=1))
        Sigma = B @ Lambda @ B.T + D

        try:
            weights = np.linalg.solve(Sigma, raw_mean)
        except np.linalg.LinAlgError:
            weights = np.ones(len(names))

        weights = np.clip(weights, 0, None)
        if weights.sum() == 0:
            weights = np.ones(len(names))
        weights = weights / weights.sum()
        weights = np.minimum(weights, self.p.max_stock_weight)
        weights = weights / weights.sum()

        # Apply turnover limit
        turnover = sum(abs(weights[i] - self.current_alloc.get(names[i], 0)) for i in range(len(names)))
        if turnover > self.p.turnover_limit:
            scale = self.p.turnover_limit / turnover
            weights = np.array([self.current_alloc.get(names[i], 0) + scale * (weights[i] - self.current_alloc.get(names[i], 0)) for i in range(len(names))])
            weights = weights / weights.sum()

        # Reset target weights
        for data in self.datas:
            self.target_alloc[data] = 0
        for w, data in zip(weights, names):
            self.target_alloc[data] = w

    # ------------------------------------------------------------------
    def _submit_orders(self, current_date):
        # Reuse sell-then-buy logic from the previous strategy
        self.sell_alloc = {}
        for data in self.stock_datas:
            current_weight = self.current_alloc[data]
            target_weight = self.target_alloc[data]
            if current_date in outside_bounds_dates:
                target_weight = (current_weight + target_weight) / 2
            cut_weight = current_weight - target_weight
            adjusted_target_weight = current_weight - cut_weight
            if current_weight > target_weight:
                self.sell_alloc[data] = adjusted_target_weight
                self.order_target_percent(data, target=adjusted_target_weight, exectype=bt.Order.Market)
        self.buy_alloc = {}
        for data in self.stock_datas:
            if data not in self.sell_alloc:
                current_weight = self.current_alloc[data]
                target_weight = self.target_alloc[data]
                if current_date in outside_bounds_dates:
                    target_weight = (current_weight + target_weight) / 2
                additional_weight = target_weight - current_weight
                if current_date in short_ma_under_long_ma_dates:
                    additional_weight = 0
                adjusted_target_weight = current_weight + additional_weight
                if adjusted_target_weight > 0:
                    self.order_target_percent(data, target=adjusted_target_weight, exectype=bt.Order.Market)
                    self.buy_alloc[data] = adjusted_target_weight


# Backwards compatibility
TestStrategy = SVDMomentumStrategy
