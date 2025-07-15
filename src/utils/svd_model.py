"""Utilities for constructing an SVD factor model."""

import numpy as np


def prepare_returns(data_feeds, lookback=252):
    """Return a demeaned and winsorized return matrix.

    Parameters
    ----------
    data_feeds : Iterable
        Backtrader data feeds providing ``close`` prices.
    lookback : int, optional
        Number of trading days to look back. Defaults to 252.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(lookback, N)`` of winsorized excess returns.
    """
    returns = []
    for feed in data_feeds:
        daily = [feed.close[-i] / feed.close[-i - 1] - 1 for i in range(1, lookback + 1)]
        returns.append(daily[::-1])
    R = np.asarray(returns).T

    R_tilde = R - R.mean(axis=0)
    for i in range(R_tilde.shape[1]):
        low, high = np.quantile(R_tilde[:, i], [0.01, 0.99])
        R_tilde[:, i] = np.clip(R_tilde[:, i], low, high)
    return R_tilde


def compute_factor_model(R_tilde, k):
    """Compute factor model parameters using SVD.

    Parameters
    ----------
    R_tilde : numpy.ndarray
        Demeaned and winsorized return matrix of shape ``(T, N)``.
    k : int
        Number of factors to retain.

    Returns
    -------
    tuple of numpy.ndarray
        Factor loadings ``B`` (``N``x``k``), factor covariance ``Lambda``
        (``k``x``k``), and idiosyncratic variances as a 1-D array.
    """
    u, s, vt = np.linalg.svd(R_tilde, full_matrices=False)
    k = min(k, len(s))
    B = vt.T[:, :k]
    Lambda = np.diag((s[:k] ** 2) / (R_tilde.shape[0] - 1))
    resid = R_tilde - R_tilde @ B @ B.T
    idio_var = resid.var(axis=0, ddof=1)
    return B, Lambda, idio_var


def tangent_portfolio(mu, B, Lambda, idio_var):
    """Compute the unconstrained mean-variance weights.

    Parameters
    ----------
    mu : numpy.ndarray
        Expected returns vector of length ``N``.
    B : numpy.ndarray
        Factor loading matrix from :func:`compute_factor_model`.
    Lambda : numpy.ndarray
        Factor covariance matrix from :func:`compute_factor_model`.
    idio_var : numpy.ndarray
        Idiosyncratic variances from :func:`compute_factor_model`.

    Returns
    -------
    numpy.ndarray
        Raw portfolio weights ``Sigma^{-1} mu`` before applying caps.
    """
    Sigma = B @ Lambda @ B.T + np.diag(idio_var)
    try:
        weights = np.linalg.solve(Sigma, mu)
    except np.linalg.LinAlgError:
        weights = np.ones_like(mu, dtype=float)
    return weights

