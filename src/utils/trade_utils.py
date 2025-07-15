"""Tools for reporting allocations and capturing Backtrader metrics.

This module defines helper functions for printing the current and target
portfolio allocations along with custom Backtrader ``Analyzer`` classes used to
record portfolio metrics and executed transactions during a backtest.
"""

import backtrader as bt


def update_allocation(strategy):
    """Report the current allocation of the portfolio."""
    print("Current Allocation...")
    total_value = strategy.broker.get_value()

    for data in strategy.d_with_len:
        value = strategy.broker.get_value([data])
        weight = value / total_value if total_value > 0 else 0
        strategy.current_alloc[data] = weight
        if weight > 0:
            print("%s: Value $%.2f, Weight %.2f%%" % (data._name, value, weight * 100))

    cash = strategy.broker.get_cash()
    print("Remaining Cash: $%.2f" % cash)


def report_target_allocation(strategy):
    """Display target allocation weights."""
    print("Target Allocation...")
    total_value = strategy.broker.get_value()
    for data, target_weight in strategy.target_alloc.items():
        target_value = target_weight * total_value
        print("%s: Target Value $%.2f, Target Weight %.2f%%" % (data._name, target_value, target_weight * 100))


class PortfolioAnalyzer(bt.Analyzer):
    """Analyzer to capture daily portfolio metrics."""

    def __init__(self):
        self.portfolio_data = []
        self.previous_portfolio_value = None

    def next(self):
        if not self.strategy.inception:
            return

        current_portfolio_value = self.strategy.broker.getvalue()
        current_cash = self.strategy.broker.get_cash()
        current_equity_value = current_portfolio_value - current_cash

        daily_pl = (
            current_portfolio_value - self.previous_portfolio_value
            if self.previous_portfolio_value
            else 0
        )
        daily_pl_percent = (
            (daily_pl / self.previous_portfolio_value) * 100
            if self.previous_portfolio_value
            else 0
        )
        accumulated_pl = current_portfolio_value - self.strategy.broker.startingcash
        accumulated_pl_percent = (accumulated_pl / self.strategy.broker.startingcash) * 100

        self.portfolio_data.append(
            {
                "date": self.strategy.datetime.date(0),
                "equity_value": current_equity_value,
                "cash_value": current_cash,
                "portfolio_value": current_portfolio_value,
                "equity_percent": (current_equity_value / current_portfolio_value) * 100,
                "cash_percent": (current_cash / current_portfolio_value) * 100,
                "daily_pl": daily_pl,
                "daily_pl_percent": daily_pl_percent,
                "accumulated_pl": accumulated_pl,
                "accumulated_pl_percent": accumulated_pl_percent,
            }
        )

        self.previous_portfolio_value = current_portfolio_value

    def get_analysis(self):
        return self.portfolio_data


class TransactionTracker(bt.Analyzer):
    """Analyzer to record executed transactions."""

    def __init__(self):
        self.transactions = []

    def notify_order(self, order):
        if not self.strategy.inception:
            return

        if order.status in [order.Completed]:
            direction = "BUY" if order.isbuy() else "SELL"
            self.transactions.append(
                {
                    "date": self.strategy.datetime.date(0),
                    "ticker": order.data._name,
                    "price": order.executed.price,
                    "size": order.executed.size,
                    "value": order.executed.value,
                    "commission": order.executed.comm,
                    "direction": direction,
                    "total cost": order.executed.value + order.executed.comm,
                }
            )

    def get_analysis(self):
        return self.transactions
