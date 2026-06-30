#!/usr/bin/env python3
"""Power-law / scaling estimators for the Bitcoin relations (PL-Cor)."""
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg


def loglog_ols(x, y):
    """OLS fit of a power law  y = A * x**b  via log10-linear regression.

    Returns (b, log10_A, r2):
      b        - power-law exponent (slope in log-log space)
      log10_A  - intercept = log10 of the prefactor A
      r2       - coefficient of determination in log-log space
    """
    # In log10 space the power law is linear: log10(y) = log10(A) + b*log10(x).
    lx, ly = np.log10(x), np.log10(y)
    # Least-squares regression -> slope b and intercept log10(A).
    b, log10_A = np.polyfit(lx, ly, 1)
    # R^2 = 1 - SS_res / SS_tot, evaluated on the log-transformed residuals.
    resid = ly - (b * lx + log10_A)
    r2 = 1.0 - np.sum(resid**2) / np.sum((ly - ly.mean())**2)
    return b, log10_A, r2


def loglog_quantile(x, y, q):
    """Quantile regression of log10(y) on log10(x) at quantile q in (0, 1).

    Returns (b, log10_A) of the q-quantile power law  y = A * x**b  -- e.g.
    the 'PL bottom' line below which a fraction q of the points lie
    (same method as price_osci's fit_power_law_bottom).
    """
    lx, ly = np.log10(x), np.log10(y)
    X = sm.add_constant(lx)                       # design matrix [1, log10 x]
    res = QuantReg(ly, X).fit(q=q)                # minimise q-weighted abs error
    log10_A, b = res.params[0], res.params[1]
    return b, log10_A
