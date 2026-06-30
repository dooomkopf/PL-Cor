stationarity.py -- Unit-root / stationarity tests for the Bitcoin scaling laws
==============================================================================

IDEA
----
Regressing two trending (non-stationary) series can give a SPURIOUS fit. The
clean way to tell a genuine relation from a spurious one is to test whether the
regression RESIDUAL is stationary (mean-reverting) or has a unit root (drifts away).

This script runs that test on two residuals (plus, for context, the raw
log-levels and their first differences). The residuals are defined EXPLICITLY --
both in LOG10 space, relative to the ORDINARY-LEAST-SQUARES MEAN power-law fit
(loglog_ols; NOT the PL-bottom quantile), with NO extra affine transform
(no clipping, no normalisation):

  r_P : price vs its OWN time power law   P = A * t^b
        OLS fit:  log10(P) = log10(A) + b * log10(t)
        r_P = log10(P) - ( log10(A) + b * log10(t) )

  r_H : hashrate vs the PRICE relation    H = A' * P^gamma   (NOT H vs t)
        OLS fit:  log10(H) = log10(A') + gamma * log10(P)
        r_H = log10(H) - ( log10(A') + gamma * log10(P) )

r_P measures how far the price sits from its own trend; r_H how far the hashrate
sits from what the price predicts. They use DIFFERENT references (t for r_P,
P for r_H) -- do not conflate them.

NOTATION
--------
  OLS MEAN fit  = ordinary-least-squares fit in log10-log10 space = the MEAN
                  trend line through the data. (Distinct from the PL-BOTTOM
                  quantile fit -- a support line below the cloud, used in other
                  scripts -- which is NOT used here.)
  I(0) / I(1)   = integration order: I(0) = stationary, I(1) = one unit root.

MATH / METHOD
-------------
Two complementary unit-root tests per series:

  ADF  (Augmented Dickey-Fuller)   H0: unit root (NON-stationary)
                                   p < ALPHA  -> reject -> stationary
  KPSS (Kwiatkowski-P-S-Shin)      H0: stationary
                                   p < ALPHA  -> reject -> NON-stationary

Because the null hypotheses are swapped, reading both gives a robust verdict:
  ADF stationary AND KPSS stationary -> I(0)  (stationary)
  ADF unit root  AND KPSS unit root  -> I(1)  (one difference needed)
  otherwise                          -> ambiguous

A STATIONARY residual (I(0)) means the power law is a genuine, cointegrating
relation -- the two series are tied by a mean-reverting "leash". A residual
with a unit root means the fit is spurious. (Same logic as the Engle-Granger
test, here applied directly to the fit residuals.)

HOW TO READ THE OUTPUT
----------------------
Read the verdict column per series: an I(0) residual indicates a genuine
(cointegrating) relation; an I(1) residual indicates the fit is spurious in
level space. Whichever way r_P and r_H come out on the current data, the script
only reports the test -- it does not assert a conclusion here.

USAGE
-----
  python stationarity.py
No CLI options. Edit the parameter block at the top of the file to change:
  ALPHA       significance level (default 0.05)
  REGRESSION  'c' = constant only (no deterministic trend)
  KPSS_NLAGS  long-run-variance bandwidth ('auto')
  INCLUDE_N   also test log10(active addresses)

OUTPUT
------
- a table (stdout) of ADF/KPSS statistics, p-values and the I(0)/I(1) verdict
- stationarity.png: two panels of r_P and r_H vs network age (log x-axis)

DEPENDENCIES
------------
numpy, matplotlib, statsmodels  (+ local modules data_io.py, scaling.py)
Needs cm_data.csv -- run  python fetch_coinmetrics.py  first.
