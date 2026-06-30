#!/usr/bin/env python3
"""Unit-root / stationarity tests for the Bitcoin scaling-law series.

We run TWO complementary tests on each (log10) series:

  ADF  (Augmented Dickey-Fuller)
      H0: a unit root is present (series is NON-stationary).
      p < ALPHA  -> reject H0 -> evidence FOR stationarity.

  KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
      H0: the series IS (level-) stationary.
      p < ALPHA  -> reject H0 -> evidence AGAINST stationarity.

The two have swapped null hypotheses, so reading them together gives a
robust integration-order verdict:

  ADF stationary AND KPSS stationary  -> I(0)  (stationary)
  ADF unit root  AND KPSS unit root   -> I(1)  (needs one difference)
  anything else                       -> ambiguous

HEADLINE TEST -- residuals (deviation from the fitted power law).

EXPLICIT RESIDUAL DEFINITION (so it is unambiguous what is tested):
  - both residuals live in LOG10 space,
  - relative to the ORDINARY-LEAST-SQUARES MEAN power-law fit (loglog_ols),
    NOT the PL-bottom quantile fit (loglog_quantile) used elsewhere,
  - with NO additional affine transformation beyond the fit's own (intercept,
    slope); no clipping, no normalisation.

  P ~ t  mean fit:  log10(P) = log10(A) + b*log10(t)
      r_P = log10(P) - ( log10(A) + b*log10(t) )

  H ~ P  mean fit:  log10(H) = log10(A') + gamma*log10(P)
      r_H = log10(H) - ( log10(A') + gamma*log10(P) )

A STATIONARY residual means the regression is a genuine (cointegrating)
relation -- the two series are tied by a mean-reverting "leash". A residual
with a unit root means the fit is SPURIOUS. This is the same logic the
Engle-Granger test in cointegration.py applies to r_P and r_H.

We additionally report the raw levels (expected I(1)) and their first
differences (expected I(0)) as context.
"""
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

from data_io import load_cm, positive, days_since_genesis
from scaling import loglog_ols

# ------------------------------ parameters -------------------------------
HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_FILE   = os.path.join(HERE, 'cm_data.csv')
STYLE_FILE  = os.path.join(HERE, 'public.mplstyle')
PLOT_FILE   = os.path.join(HERE, 'stationarity.png')
ALPHA       = 0.05          # significance level for both tests
REGRESSION  = 'c'           # 'c' = constant only, no deterministic trend
KPSS_NLAGS  = 'auto'        # data-driven bandwidth for the KPSS long-run var
INCLUDE_N   = True          # also test log10(N) (active addresses)
BTC_COLOR   = '#ff9e3d'     # orange scatter for the BTC residual points
FG_COLOR    = '#ecdcc4'     # light foreground (zero line, suptitle)
# -------------------------------------------------------------------------


def run_adf(series):
    """Augmented Dickey-Fuller test. Returns (stat, p).

    H0: unit root (non-stationary). Lag order is chosen by AIC (autolag).
    """
    stat, p, *_ = adfuller(series, regression=REGRESSION, autolag='AIC')
    return stat, p


def run_kpss(series):
    """KPSS test. Returns (stat, p).

    H0: stationary. KPSS warns (InterpolationWarning) when the statistic
    falls outside its critical-value table, i.e. the true p-value is more
    extreme than the reported bound; we silence that purely-cosmetic warning.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')           # hide InterpolationWarning
        stat, p, *_ = kpss(series, regression=REGRESSION, nlags=KPSS_NLAGS)
    return stat, p


def verdict(adf_p, kpss_p):
    """Combine the two tests into an integration-order label.

    ADF rejects unit root (adf_p < ALPHA) -> stationary signal.
    KPSS rejects stationarity (kpss_p < ALPHA) -> non-stationary signal.
    """
    adf_stationary  = adf_p  < ALPHA          # ADF says: no unit root
    kpss_stationary = kpss_p > ALPHA          # KPSS says: stationary
    if adf_stationary and kpss_stationary:
        return 'I(0)'
    if (not adf_stationary) and (not kpss_stationary):
        return 'I(1)'
    return 'ambiguous'


def pl_mean_residual(P, t):
    """Deviation of log10(P) from the OLS MEAN power law  P = A * t**b.

    Explicit:  r_P = log10(P) - (log10A + b*log10 t),  in LOG10 space, relative
    to the OLS MEAN fit (loglog_ols) -- NOT the PL-bottom quantile -- with no
    extra affine transform. A stationary r_P => the P~t power law is genuine.
    """
    b, log10_A, _ = loglog_ols(t, P)
    return np.log10(P) - (log10_A + b * np.log10(t))


def pl_relation_residual(H, P):
    """Deviation of log10(H) from the OLS MEAN relation  H = A * P**gamma.

    Explicit:  r_H = log10(H) - (log10A + gamma*log10 P),  in LOG10 space,
    relative to the OLS MEAN fit (loglog_ols) -- NOT a quantile/bottom fit --
    with no extra affine transform. Stationary r_H => genuine (cointegrated) H~P.
    """
    gamma, log10_A, _ = loglog_ols(P, H)
    return np.log10(H) - (log10_A + gamma * np.log10(P))


def plot_residuals(t, r_P, r_H, p_P, p_H):
    """Two stacked panels of the residuals vs network age (x = log days).

    A stationary residual oscillates around 0 with bounded amplitude (the
    mean-reverting "leash"); a non-stationary one drifts away from 0.
    """
    plt.style.use(STYLE_FILE)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 9))
    panels = [(ax_top, r_P, p_P, 'P-t'), (ax_bot, r_H, p_H, 'H-P')]
    for ax, resid, p, label in panels:
        ax.scatter(t, resid, s=6, alpha=0.6, color=BTC_COLOR)
        ax.axhline(0.0, color=FG_COLOR, lw=1.0)
        ax.set_xscale('log')
        ax.set_title(f'{label} residual  (ADF p={p:.3f})')
        ax.set_xlabel('days since genesis')
        ax.set_ylabel('log10 deviation from fit')
    fig.suptitle('Residual Stationarity (deviation from fit)', color=FG_COLOR)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    plt.show()


def build_series(P, H, N, t):
    """Assemble the (name, values) list of log10 series to be tested.

    Residual tests are the headline; raw levels and first differences of the
    stochastic price/hashrate follow as context. All share the same days.
    """
    lP, lH, lt = np.log10(P), np.log10(H), np.log10(t)
    series = [
        ('r_P (P~t)',     pl_mean_residual(P, t)),
        ('r_H (H~P)',     pl_relation_residual(H, P)),
        ('log10(P)',      lP),
        ('log10(H)',      lH),
        ('log10(t)',      lt),
    ]
    if INCLUDE_N:
        series.append(('log10(N)', np.log10(N)))
    series += [
        ('diff log10(P)', np.diff(lP)),
        ('diff log10(H)', np.diff(lH)),
    ]
    return series


def main():
    d = load_cm(DATA_FILE)
    t_age = days_since_genesis(d['date'])
    if INCLUDE_N:
        P, H, N, t = positive(d['P'], d['H'], d['N'], t_age)
    else:
        P, H, t = positive(d['P'], d['H'], t_age)
        N = None

    series = build_series(P, H, N, t)

    print(f"Stationarity tests  (regression='{REGRESSION}', "
          f"alpha={ALPHA}, N={len(P)} days)")
    print(f"  ADF  H0: unit root (non-stationary)  -> p<{ALPHA}: stationary")
    print(f"  KPSS H0: stationary                  -> p<{ALPHA}: non-stationary")
    print(f"  r_P, r_H = residuals from the PL fits "
          f"(headline: stationary = genuine relation)")
    print()
    header = (f"{'series':16s} {'ADF stat':>9s} {'ADF p':>7s} "
              f"{'KPSS stat':>10s} {'KPSS p':>7s}   verdict")
    print(header)
    print('-' * len(header))
    adf_p = {}
    for name, x in series:
        a_stat, a_p = run_adf(x)
        k_stat, k_p = run_kpss(x)
        adf_p[name] = a_p
        print(f"{name:16s} {a_stat:9.3f} {a_p:7.4f} "
              f"{k_stat:10.3f} {k_p:7.4f}   {verdict(a_p, k_p)}")

    print()
    print("Note: log10(t) is a deterministic monotone trend, not a stochastic "
          "process;\n      its ADF result is only illustrative.")

    plot_residuals(t, series[0][1], series[1][1],
                   adf_p['r_P (P~t)'], adf_p['r_H (H~P)'])


if __name__ == '__main__':
    main()
