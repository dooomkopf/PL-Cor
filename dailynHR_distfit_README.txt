dailynHR_distfit.py -- Noise law of the daily hashrate exponent n_HR
=====================================================================

IDEA
----
Every estimator in this repo that carries an uncertainty band needs to know
what the NOISE of the daily exponents actually looks like. Instead of assuming
Gaussian errors (or re-fitting a noise scale inside each model), we MEASURE the
noise law ONCE, in exponent space, per halving cycle -- and then treat it as a
KNOWN, FIXED input everywhere downstream. This script is that measurement for
the hashrate exponent n_HR.

Exponent space is the right place to do this: the daily exponent
n_HR(t) = log(H_2/H_1) / log(t_2/t_1) (see dailynHR.daily_exponent) removes
the power-law trend, so what remains is (approximately) stationary fluctuation
noise whose distribution can be fitted honestly. Location AND scale drift per
halving cycle, so the per-cycle fit is the honest one; a global fit is shown
only for contrast.

MATH / METHOD
-------------
Candidates fitted by maximum likelihood, per cycle and globally:
Normal, Laplace, Student-t, Generalized Normal (Subbotin; shape beta
interpolates Laplace(1) <-> Normal(2)) and Cauchy.

Model comparison on three axes:
  - overall:      log-likelihood, AIC/BIC
  - center/peak:  Kolmogorov-Smirnov (KS)
  - tails:        Anderson-Darling (tail-sensitive)

RESULT (this is where the ssmix constants come from):
Student-t wins per cycle (AIC): '13 (sigma=219, nu=17.5) - '17 (377, 9.2) -
'21 (566, 21.8) - '25 (697, 158.8). These values are hardcoded as HASH_CYC in
ssmix_HP.py and consumed by robust_smoother (band) and ssmix_TVP
(tvp_gamma R_t; level_tvp_kalman noise='ours'). They are never re-fitted
in-model. The analogous price-exponent noise is Laplace per cycle (PRICE_CYC).

USAGE
-----
  python dailynHR_distfit.py            # tables + both figures
  python dailynHR_distfit.py --no-plot  # tables only

OUTPUT
------
- per-cycle and global fit tables with AIC/BIC/KS/AD winners (stdout)
- dailynHR_distfit_cycles.png: per-cycle PDFs (log-y) with the two best laws
- dailynHR_distfit_global.png: the 4 cycle Gaussians under one global
  Laplace (Huber location) -- shows why a single global scale would be wrong

DEPENDENCIES
------------
numpy, scipy, matplotlib  (+ local modules data_io.py, dailynHR.py)
Needs cm_data.csv -- run  python fetch_coinmetrics.py  first.
