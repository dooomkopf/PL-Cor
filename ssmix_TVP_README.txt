ssmix_TVP -- M2: time-varying coupling gamma(t) between price and hashrate
=========================================================================

RED THREAD -- the idea, step by step, reduced to the essentials.

GOAL
----
How strongly are the price exponent n_P and the hashrate exponent n_H coupled,
and how does that change over time?   H ~ P^gamma   <=>   n_H = gamma * n_P.

We work in EXPONENT space because there our noise is STATIONARY and MEASURED
(Laplace for price / Student-t for hashrate, per halving cycle). The level space
(ln H vs ln P) is I(1) -> spurious regression; the stationary exponent space
complements that level-space analysis.


STEP 1 -- signals          (ssmix_TVP_methods.load_signals)
  z_P, z_H = SG-365 of the exponents = the SIGNAL (fixed; NOT re-estimated).
  sz_P, sz_H = their uncertainties, from the MEASURED per-cycle noise channel.
  Also the smoothed log-levels ln P, ln H (for the original-space comparison).

STEP 2 -- OUR coupling, EXPONENT space      (tvp_gamma)            -> PLOT 1
  Time-varying-parameter Kalman regression  z_H = gamma_exp(t) * z_P + eps.
  gamma_exp(t) is a smooth latent (poly order 1/2). Errors-in-variables: the
  observation variance R(t) = sz_H^2 + gamma^2 sz_P^2 uses the per-cycle noise.
  -> gamma_exp(t) with a posterior band. THIS IS OUR METHOD.

STEP 3 -- level-space quantity, ORIGINAL space   (rolling_slope)    -> PLOT 2
  Rolling-OLS slope of the actual ln H on ln P = the level-space gamma
  (static log-log OLS slope ~ 2). Descriptive only (levels are I(1)); shown for COMPARISON.
  (Rolling OLS is crude -> wild swings when ln P is flat in a window. A
   Kalman-smoothed level estimate (~0.82-1.63) is smoother -- see "OPTION Y" below.)

STEP 4 -- does OUR method reproduce it?   (integrate_exp_to_level) -> PLOT 2
  Integrate our exponent gamma back:  ln H_rec = sum( gamma_exp * d(ln P) ),
  then its rolling slope. If that curve sits on the actual-ln-H curve, our
  independent exponent-space estimate reproduces the level coupling.

KEY DISTINCTION (do not confuse again)
--------------------------------------
  gamma_exp  = LOCAL slope (tangent) of the log-log curve = our fluctuation coupling.
  gamma_orig = OVERALL slope (secant) of the log-log curve = the trend (static OLS ~ 2).
  They are connected by INTEGRATION (step 4), NOT by a power -> they genuinely differ
  (~0.4 vs ~2). One is not the power-transform of the other.


OPTION Y (open / pending)
-------------------------
For a FAIR comparison of gamma(t) with the level-space Kalman curve, panel (b)
should be estimated with a Kalman -- but using OUR noise, not Gaussian. Subtlety:
in the level space our noise is the INTEGRATED exponent noise = a random walk.
Differencing the level regression  d(ln H) = gamma * d(ln P) + eps  brings back
n_H = gamma * n_P -- i.e. it may reduce to our exponent-space TVP (PLOT 1).
Whether option (y) is genuinely distinct or equivalent to PLOT 1 is open.


FILES
-----
  ssmix_TVP.py            -- orchestrator (run this)
  ssmix_TVP_methods.py    -- ALL the math (no plotting)
  ssmix_TVP_plot1.py      -- plot (a) exponent-space gamma_exp(t)  [OUR method]
  ssmix_TVP_plot2.py      -- plot (b) original-space: actual vs our integrated
  ssmix_model.md          -- the full ssmix concept (signal + measured noise)

RUN:  python ssmix_TVP.py [--SG 365] [--poly 2] [--bwg 365] [--win 365]
