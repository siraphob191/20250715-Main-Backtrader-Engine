import pandas as pd
import config


def load_sector_library(path=None):
    """Load the sector library CSV and return the DataFrame and mapping."""
    if path is None:
        path = config.sector_library_path
    df = pd.read_csv(path)
    mapping = dict(zip(df['Ticker'], df['Sector']))
    return df, mapping


def load_stock_list(path=None):
    """Load the stock list CSV and return the DataFrame and ticker\u2013sector map."""
    if path is None:
        path = config.stock_list_path
    df = pd.read_csv(path)
    stock_sector_map = dict(zip(df['Custom_Code'], df['GICS Sector'].astype(str)))
    return df, stock_sector_map
