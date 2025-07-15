import config


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
    return {
        date: set(df.iloc[i, 1:].dropna().tolist())
        for i, date in enumerate(df[df.columns[0]])
    }
