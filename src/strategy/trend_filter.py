import pandas as pd
import matplotlib.pyplot as plt


def market_trend_filter(spx_df, short_ma_window=1, long_ma_window=200, show_plot=True):
    """Calculate market trend filter using moving averages.

    Parameters
    ----------
    spx_df : :class:`pandas.DataFrame`
        Benchmark dataframe indexed by date with an ``Adj Close`` column.
    short_ma_window : int, optional
        Window for short moving average. Default is ``1``.
    long_ma_window : int, optional
        Window for long moving average. Default is ``200``.
    show_plot : bool, optional
        If ``True``, display a chart with the price, short MA and long MA and
        highlight up and down trend periods.

    Returns
    -------
    :class:`pandas.DatetimeIndex`
        Dates where the short moving average is below the long moving average.
    """
    spx_df = spx_df.copy()
    spx_df['Short_MA'] = spx_df['Adj Close'].rolling(window=short_ma_window).mean()
    spx_df['Long_MA'] = spx_df['Adj Close'].rolling(window=long_ma_window).mean()

    short_under_long = spx_df[spx_df['Short_MA'] < spx_df['Long_MA']].index

    if show_plot:
        fig, ax = plt.subplots()
        spx_df['Adj Close'].plot(ax=ax, label='Adj Close', alpha=0.5)
        spx_df['Short_MA'].plot(ax=ax, label='Short MA ({})'.format(short_ma_window))
        spx_df['Long_MA'].plot(ax=ax, label='Long MA ({})'.format(long_ma_window))

        up_trend = spx_df['Short_MA'] >= spx_df['Long_MA']
        ymin, ymax = spx_df['Adj Close'].min(), spx_df['Adj Close'].max()
        ax.fill_between(spx_df.index, ymin, ymax, where=up_trend, color='green', alpha=0.1, label='Up Trend')
        ax.fill_between(spx_df.index, ymin, ymax, where=~up_trend, color='red', alpha=0.1, label='Down Trend')

        ax.set_title('Market Trend Filter')
        ax.legend()
        plt.show()

    return short_under_long
