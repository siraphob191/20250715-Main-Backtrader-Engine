from .. import config


def load_sp500_by_date(path=None):
    """Return a mapping of date to active S&P 500 constituents.

    Parameters
    ----------
    path : str, optional
        CSV file path. Uses :data:`config.sp500_constituents_path` when ``None``.

    Returns
    -------
    dict
        Maps ``date`` objects to sets of tickers.
    """
    import pandas as pd

    if path is None:
        path = config.sp500_constituents_path

    df = pd.read_csv(path, parse_dates=[0])
    df[df.columns[0]] = df[df.columns[0]].dt.date

    date_col = df.columns[0]
    mapping = {}
    current = None
    start = df[date_col].min()
    end = config.BACKTEST_END_DATE.date()
    all_dates = pd.date_range(start, end)

    rows = {row[date_col]: set(row[1:].dropna().tolist()) for _, row in df.iterrows()}

    for dt_ in all_dates:
        date = dt_.date()
        if date in rows:
            current = rows[date]
        mapping[date] = current or set()

    return mapping
