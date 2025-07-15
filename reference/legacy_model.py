# In[1]:
# Import Libraries

import backtrader as bt
import backtrader.indicators as btind
import backtrader.feeds as btfeeds
import backtrader.analyzers as btanalyzers
import datetime as dt
import pyfolio as pf
import pandas as pd
import numpy as np
import collections
import uuid                                 #in case for creating unique identifier
import matplotlib.pyplot as plt
import os
import math

# In[2]:
# Define Utility Functions to detect acquisition activity within the lookback period

def is_stock_inactive_within_period(stock_data, start_date, end_date, consecutive_days=3):
    """
    Check if a stock has the same adjusted close price for a given number of consecutive days within a specified period.
    
    Parameters:
    stock_data (pd.DataFrame): DataFrame of stock prices with 'Date' and 'Adj Close'.
    start_date (datetime): Start date of the period.
    end_date (datetime): End date of the period.
    consecutive_days (int): Number of consecutive days to check for identical prices.
    
    Returns:
    bool: True if the stock is inactive within the period, False otherwise.
    """
    period_data = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]
    return (period_data['Adj Close'].rolling(window=consecutive_days).apply(lambda x: len(set(x)) == 1, raw=True).any())

# In[3]:
# Feeding the Raw Data

# Paths to Raw Data
etf_tickers = ['XLY', 'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLB', 'XLP', 'XLU', 'XLRE', 'XLC']  # List of ETF tickers to consider
etf_data_path = '/Users/siraphobpongkritsagorn/Documents/3 Resources/Historical Data/Sector ETFs Data'
stock_data_path = '/Users/siraphobpongkritsagorn/Documents/3 Resources/Historical Data/Stocks Data/Raw Data YF format (handled 0 adj close) Sample for Testing'
benchmark_data_path = '/Users/siraphobpongkritsagorn/Documents/3 Resources/Historical Data/Benchmark Data'

# Defining the model number for referencing
model_number = "4_6"

# Specify the dates for data_feeds
start_date = dt.datetime(1999, 3, 1)  # Set a consistent start date for all data
end_date = dt.datetime(2021, 12, 31)  # Set a consistent end date for all data

# Initializing data_feeds
data_feeds = []

# Function to add specific ETF data feeds
def load_etf_feeds(tickers, path, start_date, end_date):
    for ticker in tickers:
        csv_file_path = os.path.join(path, "{}.csv".format(ticker))
        data = bt.feeds.YahooFinanceCSVData(dataname=csv_file_path, fromdate=start_date, todate=end_date)
        data_feeds.append(data)

def load_stock_feeds(path, start_date, end_date, lookback_period=252):
    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            ticker = filename.split('.')[0]
            csv_file_path = os.path.join(path, "{}.csv".format(ticker))
            stock_df = pd.read_csv(csv_file_path, parse_dates=['Date'])
            
            # Check if stock is inactive within the lookback period
            lookback_start_date = end_date - pd.DateOffset(days=lookback_period)
            if not is_stock_inactive_within_period(stock_df, lookback_start_date, end_date):
                data = bt.feeds.YahooFinanceCSVData(dataname=csv_file_path, fromdate=start_date, todate=end_date)
                data_feeds.append(data)

# Function to load the market benchmark data
def load_benchmark_data(path, ticker, start_date, end_date):
    csv_file_path = os.path.join(path, "{}.csv".format(ticker))
    data = bt.feeds.YahooFinanceCSVData(dataname=csv_file_path, fromdate=start_date, todate=end_date)
    return data

# Load ETF data feeds
load_etf_feeds(etf_tickers, etf_data_path, start_date, end_date)

# Load Stocks data feeds
load_stock_feeds(stock_data_path, start_date, end_date)

# Load Benchmark data feed
benchmark_data = load_benchmark_data(benchmark_data_path, "^SP500TR", start_date, end_date)
data_feeds.append(benchmark_data)

# In[4]:
# Special Cell: When dealing with only subsets of stocks for faster testing

# List to store tickers from files
stock_tickers = []

# Iterate over files in the directory
for filename in os.listdir(stock_data_path):
    if filename.endswith('.csv'):
        # Extract ticker from file name (assuming ticker is the file name without extension)
        ticker = os.path.splitext(filename)[0]
        stock_tickers.append(ticker)

print(stock_tickers)

# In[5]:
# Special SPX Daily Return Indicator

# Parameters for SPX Daily Return Indicator
rolling_window = 252  # Rolling window for mean and standard deviation calculation (can be adjusted)
sd_range = 5  # Standard deviation range for triggering rebalance (can be adjusted)

# Load the SPX data from the specified path
spx_file_path = '/Users/siraphobpongkritsagorn/Documents/3 Resources/Historical Data/Benchmark Data/^SP500TR.csv'
spx_df = pd.read_csv(spx_file_path, parse_dates=['Date'], index_col='Date')

# Calculate SPX daily returns
spx_df['Daily_Return'] = spx_df['Adj Close'].pct_change()

# Calculate rolling mean and standard deviation of SPX daily returns
spx_df['Rolling_Mean'] = spx_df['Daily_Return'].rolling(window=rolling_window).mean()
spx_df['Rolling_Std'] = spx_df['Daily_Return'].rolling(window=rolling_window).std()

# Calculate the upper and lower bounds for the indicator
spx_df['Upper_Bound'] = spx_df['Rolling_Mean'] + (sd_range * spx_df['Rolling_Std'])
spx_df['Lower_Bound'] = spx_df['Rolling_Mean'] - (sd_range * spx_df['Rolling_Std'])

# Determine if SPX daily return is outside the indicator bounds
spx_df['Outside_Bounds'] = (spx_df['Daily_Return'] > spx_df['Upper_Bound']) | (spx_df['Daily_Return'] < spx_df['Lower_Bound'])

# Create a list of dates when SPX daily return is outside bounds
outside_bounds_dates = spx_df[spx_df['Outside_Bounds']].index.tolist()

# Filter the DataFrame to include only the rows where daily return is outside bounds
outside_bounds_df = spx_df[spx_df['Outside_Bounds']]

# Create a new DataFrame with the desired columns
result_df = pd.DataFrame({
    'Outside_Bounds_Dates': outside_bounds_df.index,
    'Rolling_Mean_Return': outside_bounds_df['Rolling_Mean'],
    '1SD_Value': sd_range * outside_bounds_df['Rolling_Std'],
    'Upper_Bound_Value': outside_bounds_df['Upper_Bound'],
    'Lower_Bound_Value': outside_bounds_df['Lower_Bound'],
    'Daily_Return': outside_bounds_df['Daily_Return']
})

# Sort the columns in the desired order
result_df = result_df[['Outside_Bounds_Dates', 'Rolling_Mean_Return', '1SD_Value', 'Upper_Bound_Value', 'Lower_Bound_Value', 'Daily_Return']]

# Reset the index of the result DataFrame
result_df.reset_index(drop=True, inplace=True)

# Print the result DataFrame
result_df

# In[6]:
# SPX Trend Filter

# Define short and long moving average windows
short_ma_window = 1  # Default short MA window
long_ma_window = 200  # Default long MA window

# Calculate moving averages
spx_df['Short_MA'] = spx_df['Adj Close'].rolling(window=short_ma_window).mean()
spx_df['Long_MA'] = spx_df['Adj Close'].rolling(window=long_ma_window).mean()

# Identify dates when Short MA is under Long MA
short_ma_under_long_ma_dates = spx_df[spx_df['Short_MA'] < spx_df['Long_MA']].index

# Optionally, print some of these dates for verification
print("\ndates when Short MA is under Long MA:")
print(short_ma_under_long_ma_dates[:])

# Chart : Comparison of short and long Moving Averages with highlighting
plt.figure(figsize=(12, 6))
plt.plot(spx_df.index, spx_df['Short_MA'], label='Short MA', color='orange')
plt.plot(spx_df.index, spx_df['Long_MA'], label='Long MA', color='green')
plt.fill_between(spx_df.index, spx_df['Short_MA'], spx_df['Long_MA'], where=(spx_df['Short_MA'] < spx_df['Long_MA']), color='red', alpha=0.3)
plt.title('Short vs Long Moving Average')
plt.xlabel('Date')
plt.ylabel('Moving Average')
plt.legend()
plt.grid(True)
plt.show()

# In[7]:
# Prepare all the libraries for S&P500 Universe and Sector Matching

"""
Prepare Sector ticker and name library
"""
# Sector library to map ETFs to GICS Sectors
sector_library_path = '/Users/siraphobpongkritsagorn/Documents/3 Resources/Historical Data/Sector Library.csv'
sector_library = pd.read_csv(sector_library_path)

# Create a dictionary to map ETF tickers to sector names
sector_mapping = dict(zip(sector_library['Ticker'], sector_library['Sector']))

"""
Identify the list of SP500 constituents over time
"""
sp500_constituents_path = '/Users/siraphobpongkritsagorn/Documents/3 Resources/Historical Data/20220402 S&P 500 Constituents Symbols.csv'
sp500_constituents = pd.read_csv(sp500_constituents_path, parse_dates=[0]) # Need to Parse the Date
# Extract only the date part from each datetime in sp500_constituents
sp500_constituents['0'] = sp500_constituents['0'].dt.date

"""
To match stock tickers with associated sector names
"""
stock_list_path = '/Users/siraphobpongkritsagorn/Documents/3 Resources/Historical Data/List of Tickers Updated with Sectors.csv'
stock_list = pd.read_csv(stock_list_path)
stock_list_dict = dict(zip(stock_list['Custom_Code'], stock_list['GICS Sector'].astype(str)))

# Display each uploaded Dataframe
print(sector_library)
print(sp500_constituents)
print(stock_list_dict)

# In[8]:
# Initial Cash and Commission Parameters

initial_cash = 100000  # Initial capital for backtesting
commission_pct = 0.000  # % Commission in decimals
slippage_pct = 0.00015  # % Slippage in decimals

# In[9]:
class TestStrategy(bt.Strategy):
    """
    Section for Defining Relevant Parameters
    """
    params = (
        ("sharpe_period_short", 63),    # 3 months
        ("sharpe_period_medium", 126),  # 6 months
        ("sharpe_period_long", 252),    # 12 months
        ("rebalance_freq", 60),         # Rebalance every 3 months (approx 12 weeks)
        ("init_rebalance_count", 58),   # Start rebalancing from the first tradable date
        ("max_no_of_sectors", 4),       # Maximum number of assets to hold
        ("max_sector_weight", 1),       # Maximum weight for a single sector
        ("qualify_pct", 0.001),           # Top percentile of stocks in each sector allocation
        ("max_stock_weight", 1),        # Maximum weight for a single stock
        ("min_no_of_stocks", 3),        # Minimum number of stocks to select from each sector
        ("asset_short_sma_period", 1),  
        ("asset_long_sma_period", 200),  
        ("drawdown_threshold", 0.10),   # 10% drawdown threshold for triggering rebalance
    )
    
    """
    Section for Initializing the Strategy-Related Tools and Indicators
    """
    def __init__(self):
        # Initialize Portfolio Tracking and other attributes
        self.rebalance_count = self.params.init_rebalance_count
        self.current_alloc = {data: 0 for data in self.datas}
        self.target_alloc = {data: 0 for data in self.datas}
        self.d_with_len = []  # For subsets of the asset universe
        self.rebalance_date = None  # Initialize the rebalance_date attribute
        self.transaction_date = None  # Initialize the transaction_date attribute (t+1 rebalance date)
        self.inception = False  # Strategy Inception Signal

        # Initialize indicators for each asset
        self.short_smas = {data: bt.indicators.SimpleMovingAverage(data.close, period=self.params.asset_short_sma_period) for data in self.datas}
        self.long_smas = {data: bt.indicators.SimpleMovingAverage(data.close, period=self.params.asset_long_sma_period) for data in self.datas}
        self.sharpe_ratios = {data: self.calculate_avg_sharpe_ratio(data) for data in self.datas}

        # Mapping ETF tickers to sectors
        self.etf_to_sector = {row['Ticker']: row['Sector'] for index, row in sector_library.iterrows()}
        
        # Preparing a set for faster membership checking
        self.sp500_by_date = {date: set(sp500_constituents.iloc[index, 1:].dropna().values.tolist())
                              for index, date in enumerate(sp500_constituents.iloc[:, 0])}
        
        # Mapping stock tickers to sectors
        self.stock_to_sector = {row['Custom_Code']: row['GICS Sector'] for index, row in stock_list.iterrows()}

        # Calculate Sharpe ratios for the benchmark data
        self.benchmark_data = benchmark_data
        self.benchmark_sharpe = self.calculate_avg_sharpe_ratio(self.benchmark_data)

        # Initialize target weights for sector ETFs and benchmark
        self.sector_etf_target_weights = {ticker: 0 for ticker in etf_tickers}
        self.benchmark_target_weight = 0

    """
    Defining Momentum Indicator Calculations
    """
    # Helper method to calculate Sharpe Ratio for a single period
    def calculate_sharpe_ratio(self, daily_returns, period):
        rolling_mean = bt.indicators.SMA(daily_returns, period=period)
        rolling_std_dev = bt.indicators.StandardDeviation(daily_returns, period=period)

        # Add a small epsilon to avoid division by zero
        epsilon = 1e-8
        return rolling_mean / (rolling_std_dev + epsilon)  # Neglecting risk-free rate for simplicity

    # Helper method to calculate Avg. Sharpe Ratios for different periods 
    def calculate_avg_sharpe_ratio(self, data):
        daily_returns = data.close / data.close(-1) - 1
        return {
            'short': self.calculate_sharpe_ratio(daily_returns, self.params.sharpe_period_short),
            'medium': self.calculate_sharpe_ratio(daily_returns, self.params.sharpe_period_medium),
            'long': self.calculate_sharpe_ratio(daily_returns, self.params.sharpe_period_long),
            'average': (self.calculate_sharpe_ratio(daily_returns, self.params.sharpe_period_short) +
                        self.calculate_sharpe_ratio(daily_returns, self.params.sharpe_period_medium) +
                        self.calculate_sharpe_ratio(daily_returns, self.params.sharpe_period_long)) / 3
        }

    """
    Section for Helper Tools for Reporting and Debugging
    """
    # Helper tool to report the allocation status            
    def update_allocation(self):
        print("Current Allocation...")
        total_value = self.broker.get_value()
        
        for data in self.d_with_len:
            value = self.broker.get_value([data])
            weight = value / total_value if total_value > 0 else 0
            self.current_alloc[data] = weight

            if weight > 0:  # Only print if the weight is greater than zero
                print("%s: Value $%.2f, Weight %.2f%%" % (data._name, value, weight * 100))
                
        cash = self.broker.get_cash()
        print("Remaining Cash: $%.2f" % cash)

    # Helper tool to report the target allocation status
    def report_target_allocation(self):
        print("Target Allocation...")
        total_value = self.broker.get_value()
        for data, target_weight in self.target_alloc.items():
            target_value = target_weight * total_value
            print("%s: Target Value $%.2f, Target Weight %.2f%%" % (data._name, target_value, target_weight * 100))

    """
    Prenext: Section for Defining the conditions to start the backtest 
    (Important for multi-assets with different starting dates)
    """
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
    
    """
    Next: Section for the Core Strategy's Logic
    """
    def next(self):
        # Identify current date
        current_date = self.datas[0].datetime.date(0)
        
        # Identify the Strategy's Inception (Useful since analyzer will use this signal to start)
        self.inception = True
        
        # For all days. Count towards the rebalance schedule
        self.rebalance_count += 1
        print("Next is called on %s" % self.datas[0].datetime.date(0))
        print("rebalance_count = %d" % self.rebalance_count)

        """
        For Rebalance Dates
        """
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

            """
            Operations on the Sector-allocation level: 
            Target weights will be used to multiply with individual stocks' target weight
            """
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
            
            """
            Operations on the individual-stock-allocation level
            """
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
            
            # Filter stock_list_dict to include only available stocks in qualifying sectors
            # Exclude inactive stocks based on the lookback period
            lookback_start_date = current_date - pd.DateOffset(days=self.params.sharpe_period_long)
            filtered_stock_list_dict = {
                ticker: sector 
                for ticker, sector in stock_list_dict.items() 
                if sector in sector_name_and_allocations
                and ticker in active_stocks
                and not is_stock_inactive_within_period(pd.read_csv(os.path.join(stock_data_path, "{}.csv".format(ticker)), parse_dates=['Date']), lookback_start_date, current_date)
            }
        
            # Initialize sector ETF and benchmark target weights to 0 before aggregation
            self.sector_etf_target_weights = {data: 0 for data in self.datas if data._name in etf_tickers}
            self.benchmark_target_weight = 0

            # Loop through each qualifying sector to get the list of stocks in the sector
            for sector_name, allocation in sector_name_and_allocations.items():
                # Make a List stocks to include only those in the current sector
                sector_stocks = [ticker for ticker, sector in filtered_stock_list_dict.items() if sector == sector_name]
                print("There are {} stocks in {}".format(len(sector_stocks), sector_name))

                # Calculate the number of stocks to select based on the qualify_pct parameter
                num_stocks_to_select = math.ceil(len(sector_stocks) * self.params.qualify_pct)
                print("{} stocks from {} are in the top {}".format(num_stocks_to_select, len(sector_stocks), self.params.qualify_pct))
                
                # Ensure at least the minimum number of stocks are selected
                num_stocks_to_select = max(num_stocks_to_select, self.params.min_no_of_stocks)
                
                """
                Ranking part: Note that we must operate on data_feeds
                """
                # Call the Data_feeds from the list of sector_stocks
                sector_stocks_data = [data for data in self.d_with_len if data._name in sector_stocks]
                
                # Rank Stocks based on the average Sharpe Ratio for stocks. Add a condition to exclude unusual values of sharpe from unusual prices (e.g. being acquired)
                ranked_sector_stocks = sorted(
                    [(data, self.sharpe_ratios[data]['average']) for data in sector_stocks_data if data in self.sharpe_ratios and self.sharpe_ratios[data]['average'] <= 10000],
                    key=lambda x: x[1][0] if x[1][0] is not None else float('-inf'),
                    reverse=True
                )[:num_stocks_to_select]

                """
                Key Ranking Mechanism : Note that now we operate on data_feeds
                """
                # Filter ranked stocks by Sharpe Ratio trend condition (average Sharpe Ratio must be positive)
                qualified_sector_stocks = [(data, sharpe_ratio) for data, sharpe_ratio in ranked_sector_stocks if sharpe_ratio > 0]
                print("Number of qualifying stocks in {} is {}".format(sector_name, len(qualified_sector_stocks)))

                # Calculate the sum of Average Sharpe Ratios of the qualified stocks
                sector_stock_total_sharpe = sum(sharpe_ratio for data, sharpe_ratio in qualified_sector_stocks)

                """
                Weighting Allocation Calculation
                """
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
                    """
                    It's stored in data_feed format
                    E.g. <AAPL Data Feed>: 0.25,  # Target allocation weight for AAPL
                    """
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
                elif data._name == "^SP500TR":
                    self.target_alloc[data] = self.benchmark_target_weight

            """
            After assigning target stocks' weights, we need to assign the rest 0
            """
            # After calculating stock_allocations, assign 0 target weight to assets not in stock_name_and_weights
            for data in self.d_with_len:
                if data not in stock_name_and_weights and data not in self.sector_etf_target_weights and data._name != "^SP500TR":
                    self.target_alloc[data] = 0

            """
            Transactions Logic
            """
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
            self.update_allocation()

# In[10]:
class PortfolioAnalyzer(bt.Analyzer):
    
    def __init__(self):
        self.portfolio_data = []
        self.previous_portfolio_value = None  # To store the portfolio value of the previous day for daily P/L calculation

    def next(self):
        # Check if the inception flag is True in the strategy
        if not self.strategy.inception:
            # If not, skip this iteration
            return
        
        current_portfolio_value = self.strategy.broker.getvalue()
        current_cash = self.strategy.broker.get_cash()
        current_equity_value = current_portfolio_value - current_cash

        # Calculate daily P/L and accumulated P/L
        daily_pl = current_portfolio_value - self.previous_portfolio_value if self.previous_portfolio_value else 0
        daily_pl_percent = (daily_pl / self.previous_portfolio_value) * 100 if self.previous_portfolio_value else 0
        accumulated_pl = current_portfolio_value - self.strategy.broker.startingcash
        accumulated_pl_percent = (accumulated_pl / self.strategy.broker.startingcash) * 100

        # Append the current portfolio state to the portfolio data list
        self.portfolio_data.append({
            'date': self.strategy.datetime.date(0),
            'equity_value': current_equity_value,
            'cash_value': current_cash,
            'portfolio_value': current_portfolio_value,
            'equity_percent': (current_equity_value / current_portfolio_value) * 100,
            'cash_percent': (current_cash / current_portfolio_value) * 100,
            'daily_pl': daily_pl,
            'daily_pl_percent': daily_pl_percent,
            'accumulated_pl': accumulated_pl,
            'accumulated_pl_percent': accumulated_pl_percent
        })

        # Update the previous portfolio value for the next calculation
        self.previous_portfolio_value = current_portfolio_value

    def get_analysis(self):
        # Return the recorded portfolio data
        return self.portfolio_data

# In[11]:
class TransactionTracker(bt.Analyzer):
    
    def __init__(self):
        self.transactions = []
        #self.tradeids = {}  # Dictionary to keep track of trade IDs and related buy orders

    def notify_order(self, order):
        # Check if the inception flag is True before recording transactions
        # This ensures transactions are only tracked when the strategy starts actual trading
        if not self.strategy.inception:
            # If inception is not True, return without recording the transaction
            return

        if order.status in [order.Completed]:
            
            if order.isbuy():
                direction = 'BUY'
                # Generate a new unique trade ID for this buy order, for example, using a UUID
                #trade_id = str(uuid.uuid4())
                #self.tradeids[order.ref] = trade_id
                
            else:
                direction = 'SELL'
                # Retrieve the existing trade ID for this sell order based on the buy order's ref
                #trade_id = self.tradeids.get(order.ref, 'UNKNOWN')
                
            # Append transaction details to the transactions list
            # Only completed orders are considered, reflecting actual executed trades
            self.transactions.append({
                'date': self.strategy.datetime.date(0), # The date of the transaction
                'ticker': order.data._name,             # Ticker symbol of the traded asset
                'price': order.executed.price,          # Execution price of the order
                'size': order.executed.size,            # The size of the order (number of shares/contracts)
                'value': order.executed.value,          # The monetary value of the order
                'commission': order.executed.comm,      # Commission paid for the order
                'direction': direction,                 # The direction of the trade, 'BUY' or 'SELL'
                'total cost': order.executed.value + order.executed.comm  # Total cost including commission
                #'tradeid': trade_id,                    # Unique identifier for the buy-sell couple

            })

    def get_analysis(self):
        # Return the recorded transactions
        return self.transactions

# In[12]:
# Setting up the engine and trigger the strategy to run


# Create a cerebro object
cerebro = bt.Cerebro()

# Assuming data_feeds is your list of data feeds
for data in data_feeds:
    cerebro.adddata(data)

# Add the strategy to Cerebro and pass the stop loss and recovery dates
cerebro.addstrategy(TestStrategy)

# Set initial cash
cerebro.broker.set_cash(initial_cash)

# Add the commission
cerebro.broker.setcommission(commission=commission_pct)

# Add slippage as a percentage of the order price
cerebro.broker.set_slippage_perc(slippage_pct, slip_open=True, slip_limit=True, slip_match=True, slip_out=True)

# Add the portfolio tracker
cerebro.addanalyzer(PortfolioAnalyzer, _name='portfolio_data')

# Add the transaction tracker
cerebro.addanalyzer(TransactionTracker, _name='transaction_data')

# Add PyFolio analyzer
cerebro.addanalyzer(btanalyzers.PyFolio, _name='pyfolio')

# Run the strategy
results = cerebro.run()

# Plot the result
#cerebro.plot()

# Construct the portfolio tracker
portfolio_data = results[0].analyzers.getbyname('portfolio_data').get_analysis()

# Construct the transaction tracker
transaction_data = results[0].analyzers.getbyname('transaction_data').get_analysis()

# Construct the PyFolio items
strat = results[0]
pyfolio_analyzer = strat.analyzers.getbyname('pyfolio')
returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()

# In[13]:
# Portfolio Report
df_portfolio_data = pd.DataFrame(portfolio_data)

# Set the 'date' column as the index
df_portfolio_data.set_index('date', inplace=True)

df_portfolio_data

# In[14]:
# Transaction Report

df_transaction_data = pd.DataFrame(transaction_data)

# Set the 'date' column as the index
df_transaction_data.set_index('date', inplace=True)

df_transaction_data

#Price is adj open with slippage
#Value is based on this Price

# To verify, on the first trade date, sum(total cost) + cash_value = initial cash_value

# In[15]:
# Benchmark

# Benchmark File Path
benchmark_file_path = '/Users/siraphobpongkritsagorn/Documents/3 Resources/Historical Data/Benchmark Data/^SP500TR.csv'

df_benchmark = pd.read_csv(benchmark_file_path)

df_benchmark

# Now prepare benchmark data to be comparable to the portfolio

# Step 1: Set 'Date' as index in df_benchmark if not already
df_benchmark['Date'] = pd.to_datetime(df_benchmark['Date'])
df_benchmark.set_index('Date', inplace=True)

# Step 2: Filter df_benchmark to match the dates in df_portfolio_data
df_benchmark_filtered = df_benchmark[df_benchmark.index.isin(df_portfolio_data.index)]

# Check if df_benchmark_filtered is empty
print("Length of df_benchmark_filtered:", len(df_benchmark_filtered))

# Proceed only if df_benchmark_filtered is not empty
if len(df_benchmark_filtered) > 0:
    # Assuming the initial investment is the same as the first 'portfolio_value' in df_portfolio_data
    initial_investment = df_portfolio_data['portfolio_value'].iloc[0]
    first_benchmark_close = df_benchmark_filtered['Adj Close'].iloc[0]

    # Calculate portfolio value for each date
    df_benchmark_filtered['portfolio_value'] = initial_investment * (df_benchmark_filtered['Adj Close'] / first_benchmark_close)

    # Calculate other columns
    df_benchmark_data = df_benchmark_filtered[['portfolio_value']].copy()
    df_benchmark_data['accumulated_pl'] = df_benchmark_data['portfolio_value'] - initial_investment
    df_benchmark_data['accumulated_pl_percent'] = (df_benchmark_data['accumulated_pl'] / initial_investment) * 100
    df_benchmark_data['daily_pl'] = df_benchmark_data['portfolio_value'].diff()
    df_benchmark_data['daily_pl_percent'] = (df_benchmark_data['daily_pl'] / df_benchmark_data['portfolio_value'].shift(1)) * 100
    df_benchmark_data.fillna(0, inplace=True)
    # ...


df_benchmark_data

# In[16]:
# Performance Report

# Portfolio Performance

# Assuming df_portfolio_data is already loaded as a DataFrame

# Define a function to calculate annualized returns
def annualized_return(df):
    total_period = (df.index[-1] - df.index[0]).days / 365.25
    ending_value = df['portfolio_value'].iloc[-1]
    starting_value = df['portfolio_value'].iloc[0]
    return ((ending_value / starting_value) ** (1 / total_period)) - 1

# Function to calculate maximum drawdown
def max_drawdown(df):
    roll_max = df['portfolio_value'].cummax()
    drawdown = df['portfolio_value']/roll_max - 1.0
    return drawdown.min()

# Renamed function to calculate Sharpe ratio
def calculate_sharpe_ratio(df):
    returns = df['portfolio_value'].pct_change()
    return returns.mean() / returns.std() * np.sqrt(252)  # Assuming 252 trading days in a year

# Calculating metrics for portfolio
total_returns_portfolio = (df_portfolio_data['portfolio_value'].iloc[-1] / df_portfolio_data['portfolio_value'].iloc[0]) - 1
annualized_returns_portfolio = annualized_return(df_portfolio_data)
max_dd_portfolio = max_drawdown(df_portfolio_data)
sharpe_ratio_portfolio = calculate_sharpe_ratio(df_portfolio_data)
std_dev_portfolio = df_portfolio_data['portfolio_value'].pct_change().std() * np.sqrt(252)

# Creating summary table for portfolio
portfolio_summary_table = pd.DataFrame({
    "Total Returns": [total_returns_portfolio],
    "Annualized Returns": [annualized_returns_portfolio],
    "Max Drawdown": [max_dd_portfolio],
    "Sharpe Ratio": [sharpe_ratio_portfolio],
    "Standard Deviation": [std_dev_portfolio]
})

# Trades Report

# Assuming df_transaction_data is already loaded as a DataFrame

# Calculate total trades
total_trades = df_transaction_data.shape[0]

# Calculate average trade size
average_trade_size = df_transaction_data['size'].abs().mean()

# Calculate total commissions paid
total_commissions_paid = df_transaction_data['commission'].sum()

# Calculate profitable and loss-making trades
profitable_trades = df_transaction_data[df_transaction_data['size'] * df_transaction_data['price'] > 0]
loss_trades = df_transaction_data[df_transaction_data['size'] * df_transaction_data['price'] < 0]

# Calculate the number and percentage of profitable and loss trades
num_profitable_trades = profitable_trades.shape[0]
percent_profitable_trades = (num_profitable_trades / total_trades) * 100
num_loss_trades = loss_trades.shape[0]
percent_loss_trades = (num_loss_trades / total_trades) * 100

# Creating summary table for transactions
transaction_summary = pd.DataFrame({
    "Total Trades": [total_trades],
    "Average Trade Size": [average_trade_size],
    "Total Commissions Paid": [total_commissions_paid],
    "Profitable Trades": [num_profitable_trades],
    "Profitable Trades %": [percent_profitable_trades],
    "Loss Trades": [num_loss_trades],
    "Loss Trades %": [percent_loss_trades]
})

# Benchmark Performance

# Assuming df_benchmark_data is already loaded as a DataFrame

# Calculating metrics for benchmark
total_returns_benchmark = (df_benchmark_data['portfolio_value'].iloc[-1] / df_benchmark_data['portfolio_value'].iloc[0]) - 1
annualized_returns_benchmark = annualized_return(df_benchmark_data)
max_dd_benchmark = max_drawdown(df_benchmark_data)
sharpe_ratio_benchmark = calculate_sharpe_ratio(df_benchmark_data)
std_dev_benchmark = df_benchmark_data['portfolio_value'].pct_change().std() * np.sqrt(252)

# Creating summary table for benchmark
benchmark_summary_table = pd.DataFrame({
    "Total Returns": [total_returns_benchmark],
    "Annualized Returns": [annualized_returns_benchmark],
    "Max Drawdown": [max_dd_benchmark],
    "Sharpe Ratio": [sharpe_ratio_benchmark],
    "Standard Deviation": [std_dev_benchmark]
})

# Print the reports
print("Portfolio Performance Summary:")
print(portfolio_summary_table)
print("\nTransaction Summary:")
print(transaction_summary)
print("\nBenchmark Performance Summary:")
print(benchmark_summary_table)

# In[17]:
# Time-series Visualizations

# Assuming df_portfolio_data and df_benchmark_data are already loaded as DataFrames
# Make sure that the index (date) column is of datetime type and sorted

# Time Series Chart for Portfolio Value and Benchmark
plt.figure(figsize=(14, 7))
plt.plot(df_portfolio_data['portfolio_value'], label='Portfolio Value')
plt.plot(df_benchmark_data['portfolio_value'], label='Benchmark Value', color='green')
plt.yscale('log')  # Setting the y-axis to a logarithmic scale
plt.title('Portfolio and Benchmark Value Over Time')
plt.xlabel('Date')
plt.ylabel('Value (Log Scale)')
plt.legend()
plt.grid(True)
plt.show()

# Function to calculate drawdown series
def calculate_drawdown_series(df):
    roll_max = df['portfolio_value'].cummax()
    drawdown = df['portfolio_value'] / roll_max - 1.0
    return drawdown

# Calculate Drawdown for Portfolio and Benchmark
df_portfolio_data['drawdown'] = calculate_drawdown_series(df_portfolio_data)
df_benchmark_data['drawdown'] = calculate_drawdown_series(df_benchmark_data)

# Drawdown Chart
plt.figure(figsize=(14, 7))
plt.plot(df_portfolio_data.index, df_portfolio_data['drawdown'], label='Portfolio Drawdown', color='darkred', alpha=0.5)
plt.plot(df_benchmark_data.index, df_benchmark_data['drawdown'], label='Benchmark Drawdown', color='darkgreen', alpha=0.5)
plt.fill_between(df_portfolio_data.index, df_portfolio_data['drawdown'], color='red', alpha=0.3)
plt.fill_between(df_benchmark_data.index, df_benchmark_data['drawdown'], color='green', alpha=0.3)
plt.title('Drawdown Over Time')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.legend()
plt.grid(True)
plt.show()

# Equity Exposure Over Time (Using 'equity_percent' Column)
plt.figure(figsize=(14, 7))
plt.fill_between(df_portfolio_data.index, df_portfolio_data['equity_percent'], 
                 label='Equity Exposure (%)', alpha=0.5, color='skyblue')
plt.title('Equity Exposure Over Time')
plt.xlabel('Date')
plt.ylabel('Equity Exposure (%)')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Daily P/L Chart for Portfolio
plt.figure(figsize=(14, 7))
plt.plot(df_portfolio_data['daily_pl_percent'], label='Portfolio Daily P/L %', color='blue')
plt.title('Portfolio Daily P/L % Over Time')
plt.xlabel('Date')
plt.ylabel('Daily P/L')
plt.legend()
plt.grid(True)
plt.show()

# Daily P/L Chart for Benchmark
plt.figure(figsize=(14, 7))
plt.plot(df_benchmark_data['daily_pl_percent'], label='Benchmark Daily P/L $', color='green')
plt.title('Benchmark Daily P/L % Over Time')
plt.xlabel('Date')
plt.ylabel('Daily P/L')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Accumulated P/L
plt.figure(figsize=(14, 7))
plt.plot(df_portfolio_data['accumulated_pl_percent'], label='Portfolio Return', color='orange')
plt.plot(df_benchmark_data['accumulated_pl_percent'], label='Benchmark Return', color='purple')
plt.title('Accumulated Return Over Time')
plt.xlabel('Date')
plt.ylabel('Accumulated Return')
plt.legend()
plt.grid(True)
plt.show()

# Histogram of Daily Returns for Portfolio
plt.figure(figsize=(14, 7))
plt.hist(df_portfolio_data['daily_pl'].dropna() * 100, bins=50, alpha=0.75, label='Portfolio')
plt.title('Histogram of Daily Returns for Portfolio')
plt.xlabel('Daily Returns (%)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Histogram of Daily Returns for Benchmark
plt.figure(figsize=(14, 7))
plt.hist(df_benchmark_data['daily_pl'].dropna() * 100, bins=50, alpha=0.75, color='green', label='Benchmark')
plt.title('Histogram of Daily Returns for Benchmark')
plt.xlabel('Daily Returns (%)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# In[18]:
# Extracting Results to .CSV

"""
# Define the base path and model number
base_path = '/Users/siraphobpongkritsagorn/Documents/Leading Assets United/Python Experiment/20240415 Backtrader Sector Stock Momentum/Results'

# Create a directory specific to the model number
model_path = os.path.join(base_path, "model{}".format(model_number))
os.makedirs(model_path, exist_ok=True)

# Extract and format commission and slippage
commission_bp = int(commission_pct * 10000)  # Convert to basis points
slippage_bp = int(slippage_pct * 10000)      # Convert to basis points
# Extract and format start date
start_date_formatted = start_date.strftime('%Y%m%d')  # Format datetime to 'YYYYMMDD'

# Save CSVs to the specified directory
df_portfolio_data.to_csv(os.path.join(model_path, 'portfolio_data_{}_{}_{}_{}.csv'.format(model_number, commission_bp, slippage_bp, start_date_formatted)), index=True)
df_transaction_data.to_csv(os.path.join(model_path, 'transaction_data_{}_{}_{}_{}.csv'.format(model_number, commission_bp, slippage_bp, start_date_formatted)), index=True)
portfolio_summary_table.to_csv(os.path.join(model_path, 'portfolio_performance_summary_{}_{}_{}_{}.csv'.format(model_number, commission_bp, slippage_bp, start_date_formatted)), index=False)
transaction_summary.to_csv(os.path.join(model_path, 'transaction_summary_{}_{}_{}_{}.csv'.format(model_number, commission_bp, slippage_bp, start_date_formatted)), index=False)
benchmark_summary_table.to_csv(os.path.join(model_path, 'benchmark_performance_summary_{}_{}_{}_{}.csv'.format(model_number, commission_bp, slippage_bp, start_date_formatted)), index=False)

"""


# In[None]:

