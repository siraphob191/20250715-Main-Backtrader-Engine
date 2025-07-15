import backtrader as bt
import pandas as pd
import os
import math
import datetime as dt
import config

from .market_signal import get_market_signals


# Mapping of S&P 500 constituents by date. ``run_backtest`` sets this
# module-level variable before creating the strategy instance.
sp500_by_date = {}

# Market signal dates used by the strategy. ``get_market_signals`` will populate
# these when the strategy instance is created.
outside_bounds_dates = []
short_ma_under_long_ma_dates = []

from .indicators import calculate_avg_sharpe_ratio, is_stock_inactive_within_period
from .config import DEFAULT_PARAMS
from utils.trade_utils import update_allocation, report_target_allocation

class TestStrategy(bt.Strategy):
    """Momentum strategy ranking sectors and stocks by Sharpe ratio."""

    params = DEFAULT_PARAMS

    def __init__(self):
        """Initialize indicators and look-up tables."""
        # Initialize Portfolio Tracking and other attributes
        self.rebalance_count = self.params.init_rebalance_count
        self.current_alloc = {data: 0 for data in self.datas}
        self.target_alloc = {data: 0 for data in self.datas}
        self.d_with_len = []  # For subsets of the asset universe
        self.rebalance_date = None  # Initialize the rebalance_date attribute
        self.transaction_date = None  # Initialize the transaction_date attribute (t+1 rebalance date)
        self.inception = False  # Strategy Inception Signal

        # Initialize indicators for each asset
        self.short_smas = {data: bt.indicators.SimpleMovingAverage(data.close,
                                    period=self.params.asset_short_sma_period)
                           for data in self.datas}
        self.long_smas = {data: bt.indicators.SimpleMovingAverage(data.close,
                                   period=self.params.asset_long_sma_period)
                          for data in self.datas}
        self.sharpe_ratios = {
            data: calculate_avg_sharpe_ratio(
                data,
                self.params.sharpe_period_short,
                self.params.sharpe_period_medium,
                self.params.sharpe_period_long,
            )
            for data in self.datas
        }
        
        # S&P 500 constituents by date provided by ``run_backtest``
        self.sp500_by_date = sp500_by_date
        self.sp500_by_date

        # Mapping ETF tickers to sectors
        self.etf_to_sector = {row['Ticker']: row['Sector'] for index, row in sector_library.iterrows()}
        self.etf_to_sector
        
        # Mapping stock tickers to sectors
        self.stock_to_sector = stock_sector_map
        self.stock_to_sector

        # Calculate Sharpe ratios for the benchmark data
        self.benchmark_data = benchmark_data
        self.benchmark_sharpe = calculate_avg_sharpe_ratio(
            self.benchmark_data,
            self.params.sharpe_period_short,
            self.params.sharpe_period_medium,
            self.params.sharpe_period_long,
        )

        # Initialize target weights for sector ETFs and benchmark
        self.sector_etf_target_weights = {ticker: 0 for ticker in etf_tickers}
        self.benchmark_target_weight = 0

        # Load market signal dates
        global outside_bounds_dates, short_ma_under_long_ma_dates
        outside_bounds_dates, short_ma_under_long_ma_dates = get_market_signals()
        self.outside_bounds_dates = outside_bounds_dates
        self.short_ma_under_long_ma_dates = short_ma_under_long_ma_dates

        # Momentum indicator calculations are implemented in asset_indicators.py

    # Helper tools for reporting and debugging are provided in ``utils.trade_utils``.
    # ``prenext`` defines the conditions to start the backtest when assets have
    # different starting dates.
    def prenext(self):
        # Populate d_with_len with assets that at least have enough data
        self.d_with_len = [d for d in self.datas if len(d) >= self.params.sharpe_period_long]
    
        # Only call next() if d_with_len is not empty
        if self.d_with_len:
            self.next()

    # Might be redundant. This is to call next once we have all assets ready. Might not be useful in this case
    def nextstart(self):
        # This is called exactly ONCE, when next is 1st called and defaults to call `next`
        #self.d_with_len = self.datas  # All data sets fulfill the guarantees now
        if self.d_with_len:
            self.next()
    
    # The ``next`` method contains the core trading logic executed each day.
    def next(self):
        # Identify current date
        current_date = self.datas[0].datetime.date(0)
        
        # Identify the Strategy's Inception (Useful since analyzer will use this signal to start)
        self.inception = True
        
        # For all days. Count towards the rebalance schedule
        self.rebalance_count += 1
        print("Next is called on %s" % self.datas[0].datetime.date(0))
        print("rebalance_count = %d" % self.rebalance_count)

        # Logic executed on scheduled rebalance dates
        # Check if it's normal rebalance schedule or outside_bound date
        if (
            self.rebalance_count % self.params.rebalance_freq == 0
            or current_date in outside_bounds_dates # SPX daily return Outliers
        ):
            # Store the rebalance date in an instance variable
            self.rebalance_date = self.datas[0].datetime.date(0)
            print("Rebalancing portfolio TARGET SETTING on date: %s" % self.rebalance_date)
            
            # Reset the target_alloc to make sure we are not left with any target weights calculated from prev rebalance
            for data in self.d_with_len:
                self.target_alloc[data] = 0

            # ----- Sector allocation phase -----
            # Create a subset of self.d_with_len that are part of etf_tickers only
            sector_etfs = [data for data in self.d_with_len if data._name in etf_tickers]
            
            # Rank Sectors based on the average Sharpe Ratio for ETFs only
            ranked_sectors = sorted(
                [(data, self.sharpe_ratios[data]['average']) for data in sector_etfs if data in self.sharpe_ratios], 
                key=lambda x: x[1][0] if x[1][0] is not None else float('-inf'), 
                reverse=True)[:self.params.max_no_of_sectors]
            
            # Filter ranked sectors by Sharpe Ratio trend condition (average Sharpe Ratio must be positive)
            qualified_sectors = [(data, sharpe_ratio) for data, sharpe_ratio in ranked_sectors if sharpe_ratio > 0]
            
            # Calculate the sum of Average Sharpe Ratios of the qualified sectors
            sector_total_sharpe = sum(sharpe_ratio for data, sharpe_ratio in qualified_sectors)
            
            # Initialize a dictionary to store sector allocations
            sector_name_and_allocations = {}
            
            # Calculate the target allocation for each qualified sector
            for data, sharpe_ratio in qualified_sectors:
                num_qualified_sectors = len(qualified_sectors)

                # Calculate target allocation based on the sector's proportion of the total Sharpe Ratio
                target_sector_alloc = (sharpe_ratio / sector_total_sharpe) * (num_qualified_sectors / self.params.max_no_of_sectors)

                # Store the sector name and its corresponding allocation in the dictionary
                sector_name = sector_mapping.get(data._name, 'Unknown')
                sector_name_and_allocations[sector_name] = target_sector_alloc
            
            # Output will be a dict of { sector_name: Allocation }
            print("Sector Allocations:", sector_name_and_allocations) # It's a dictionary of (sector ticker : weight)
            
            # ----- Stock selection phase -----
            # Generate a list of stocks with available data in case we use a subset of stocks to test
            available_stocks = stock_tickers
            
            # Report the number of stocks in the universe (SP500 constituents by date)
            active_universe = list(set(self.sp500_by_date.get(current_date, set())))
            print("There are {} SP500 stocks on {}".format(len(active_universe), current_date))
            
            # Filter only available stocks that match the S&P 500 constituents on the rebalance date and are available
            active_stocks = list(set(self.sp500_by_date.get(current_date, set())) & set(available_stocks))
            print("Active Stocks on {}: {}".format(current_date, len(active_stocks)))
            
            # Initialize a dictionary to store stock allocations
            stock_name_and_weights = {}
            
            # Filter stock_sector_map to include only available stocks in qualifying sectors
            # Exclude inactive stocks based on the lookback period
            lookback_start_date = current_date - pd.DateOffset(days=self.params.sharpe_period_long)
            filtered_stock_sector_map = {
                ticker: sector
                for ticker, sector in stock_sector_map.items()
                if sector in sector_name_and_allocations
                and ticker in active_stocks
                and not is_stock_inactive_within_period(pd.read_csv(os.path.join(stock_data_path, "{}.csv".format(ticker)), parse_dates=['Date']), lookback_start_date, current_date)
            }
        
            # Initialize sector ETF and benchmark target weights to 0 before aggregation
            self.sector_etf_target_weights = {data: 0 for data in self.datas if data._name in etf_tickers}
            self.benchmark_target_weight = 0

            # Loop through each qualifying sector to get the list of stocks in the sector
            for sector_name, allocation in sector_name_and_allocations.items():
                # Make a list of stocks to include only those in the current sector
                sector_stocks = [ticker for ticker, sector in filtered_stock_sector_map.items() if sector == sector_name]
                print("There are {} stocks in {}".format(len(sector_stocks), sector_name))

                # Calculate the number of stocks to select based on the qualify_pct parameter
                num_stocks_to_select = math.ceil(len(sector_stocks) * self.params.qualify_pct)
                print("{} stocks from {} are in the top {}".format(num_stocks_to_select, len(sector_stocks), self.params.qualify_pct))
                
                # Ensure at least the minimum number of stocks are selected
                num_stocks_to_select = max(num_stocks_to_select, self.params.min_no_of_stocks)
                
                # Rank stocks using price data feeds
                # Call the Data_feeds from the list of sector_stocks
                sector_stocks_data = [data for data in self.d_with_len if data._name in sector_stocks]
                
                # Rank Stocks based on the average Sharpe Ratio for stocks. Add a condition to exclude unusual values of sharpe from unusual prices (e.g. being acquired)
                ranked_sector_stocks = sorted(
                    [(data, self.sharpe_ratios[data]['average']) for data in sector_stocks_data if data in self.sharpe_ratios and self.sharpe_ratios[data]['average'] <= 10000],
                    key=lambda x: x[1][0] if x[1][0] is not None else float('-inf'),
                    reverse=True
                )[:num_stocks_to_select]

                # Key ranking mechanism operates on data feeds
                # Filter ranked stocks by Sharpe Ratio trend condition (average Sharpe Ratio must be positive)
                qualified_sector_stocks = [(data, sharpe_ratio) for data, sharpe_ratio in ranked_sector_stocks if sharpe_ratio > 0]
                print("Number of qualifying stocks in {} is {}".format(sector_name, len(qualified_sector_stocks)))

                # Calculate the sum of Average Sharpe Ratios of the qualified stocks
                sector_stock_total_sharpe = sum(sharpe_ratio for data, sharpe_ratio in qualified_sector_stocks)

                # ----- Weight calculation -----
                # Calculate the target allocation for each qualified stock
                for data, sharpe_ratio in qualified_sector_stocks:
                    num_qualified_sector_stocks = len(qualified_sector_stocks)

                    # Calculate target allocation proportion of the total Sharpe Ratio
                    target_weight = (sharpe_ratio / sector_stock_total_sharpe) * (num_qualified_sector_stocks / len(ranked_sector_stocks)) * allocation * 0.99

                    # Determine if the stock's Sharpe ratio is lower than its sector ETF or the market benchmark
                    sector_etf_ticker = None
                    sector_etf_data = None
                    for etf, sector in self.etf_to_sector.items():
                        if sector == sector_name:
                            sector_etf_ticker = etf  # we get the ticker like XLK
                            sector_etf_data = next((etf_data for etf_data in sector_etfs if etf_data._name == sector_etf_ticker), None)
                            print("Sector ETF Ticker for sector {}: {}".format(sector_name, sector_etf_ticker))
                            break

                    sector_etf_sharpe = self.sharpe_ratios[sector_etf_data]['average'] if sector_etf_data else 0

                    max_sharpe = max(sharpe_ratio, sector_etf_sharpe, self.benchmark_sharpe['average'])

                    if max_sharpe == sector_etf_sharpe:
                        # Aggregate weight to the sector ETF
                        self.sector_etf_target_weights[sector_etf_data] += target_weight
                    elif max_sharpe == self.benchmark_sharpe['average']:
                        # Aggregate weight to the benchmark
                        self.benchmark_target_weight += target_weight
                    else:
                        # Assign weight to the stock itself
                        self.target_alloc[data] = min(target_weight, self.params.max_stock_weight)
                        print("{} target weight: {}".format(data._name, target_weight))
                        
                    # Store the ticker and its allocation in stock_name_and_weights 
                    # Stored in ``data_feed`` format, e.g. ``<AAPL Data Feed>: 0.25``
                    stock_name_and_weights[data] = target_weight

            # Print the aggregated target weights for sector ETFs and the benchmark
            # print("Sector ETF Target Weights:", self.sector_etf_target_weights)
            print("Benchmark Target Weight:", self.benchmark_target_weight)

            # Set target allocations for sector ETFs and benchmark
            for data in self.d_with_len:
                if data in self.sector_etf_target_weights:
                    print(len(sector_etfs))
                    print(data._name)
                    print("{} target weight : {}".format(data._name , self.sector_etf_target_weights[data]))
                    self.target_alloc[data] = self.sector_etf_target_weights[data]
                elif data._name == config.BENCHMARK_SYMBOL:
                    self.target_alloc[data] = self.benchmark_target_weight

            # After assigning weights to target stocks, everything else goes to zero
            # After calculating stock_allocations, assign 0 target weight to assets not in stock_name_and_weights
            for data in self.d_with_len:
                if (
                    data not in stock_name_and_weights
                    and data not in self.sector_etf_target_weights
                    and data._name != config.BENCHMARK_SYMBOL
                ):
                    self.target_alloc[data] = 0

            # ----- Transaction execution -----
            # Selling Phase: Important to sell first.
            self.sell_alloc = {}
            for data in self.d_with_len:
                current_weight = self.current_alloc[data] # Default starts with 0 for current_alloc
                target_weight = self.target_alloc[data]
                
                # Modify the target_weight to incorporate half rebalance SPX daily return Outliers
                if current_date in outside_bounds_dates: # SPX daily return Outliers
                    target_weight = (current_weight + target_weight) / 2
                
                cut_weight = current_weight - target_weight
                
                adjusted_target_weight = current_weight - cut_weight

                if current_weight > target_weight:
                    self.sell_alloc[data] = adjusted_target_weight
                    self.order_target_percent(data, target=adjusted_target_weight, exectype=bt.Order.Market) # Execute at next Open (adj)
                    print("Selling %s: Target %.2f%%, Current %.2f%%" % (data._name, target_weight * 100, current_weight * 100))
                    
            # Buying Phase
            self.buy_alloc = {}
            for data in self.d_with_len:
                if data not in self.sell_alloc: # We exclude assets that were just sold
                    current_weight = self.current_alloc[data]
                    target_weight = self.target_alloc[data]
                    
                    # Modify the target_weight to incorporate half rebalance SPX daily return Outliers
                    if current_date in outside_bounds_dates: # SPX daily return Outliers
                        target_weight = (current_weight + target_weight) / 2
                            
                    additional_weight = target_weight - current_weight  # Intended additional weight
                    
                    # Apply trend SPX trendfilter
                    if current_date in short_ma_under_long_ma_dates: 
                        additional_weight = additional_weight * 0 # Stop/Reduce buying when SPX short MA is under long MA
                    
                    adjusted_target_weight = current_weight + additional_weight
                    
                    if adjusted_target_weight > 0:  # Only report for assets that we want to buy
                        self.order_target_percent(data, target=adjusted_target_weight, exectype=bt.Order.Market) # Execute at next Open (adj)
                        self.buy_alloc[data] = adjusted_target_weight
                        print("Buying %s: Target %.2f%%" % (data._name, adjusted_target_weight * 100))
                        
        # For Transactions Days (The days the actual transactions happen)
        # Important to also check rebalance_date is not none
        # We check to locate the date after rebalancing date
        if self.rebalance_date and self.datas[0].datetime.date(0) == self.rebalance_date + dt.timedelta(days=1):
            # Store the transaction date in an instance variable
            self.transaction_date = self.datas[0].datetime.date(0)
            print("Transactions EXECUTED on date: %s" % self.transaction_date)
            print("Post Transactions Allocation Below")
            # Update Current Allocation 
            update_allocation(self)


