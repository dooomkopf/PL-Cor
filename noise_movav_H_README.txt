noise_movav_H.py -- MC noise-strip around a moving average of the hashrate
===========================================================================

IDEA
----
dailynHR_distfit.py measures the noise LAW of the daily hashrate exponent
(Student-t per halving cycle). This script answers the follow-up question:
does that exponent-space noise law actually reproduce the scatter of the REAL
hashrate in PRICE SPACE (the raw ln H series)? If yes, the measured law can be
carried into price space and used there as a known, fixed noise input.

The construction separates signal and noise:
  - anchor (the "signal" stand-in): a centered WINDOW-day moving average of
    the REAL hashrate -- invisible in the final plot, it only anchors the band;
  - noise: Monte-Carlo paths whose daily steps contain ONLY the measured
    exponent noise (drift-free, mu = 0 -- only the WIDTH of the law is used).

MATH / METHOD
-------------
Daily local exponent (H ~ t^n):     n_HR = ln(H_t/H_{t-1}) / ln(t/(t-1))
One MC step (exact log time factor): ln H_t = ln H_{t-1} + n * ln(t/(t-1)),
                                     n ~ Student-t(nu_c) * s_c   (per cycle c)
Per cycle c: (nu_c, s_c) = MLE Student-t fit of the daily n_HR (mu = 0).
From each anchor day, N paths run HORIZON days forward; the band is the
[100-Q .. Q] percentile range of the ensemble. A good noise law => the
Q1..Q99 band envelopes the real hashrate scatter around the anchor.

WORKFLOW / where this sits in the chain:
dailynHR_distfit.py (measure the law per cycle) -> fixed tables in
ssmix_HP.py (PRICE_CYC / HASH_CYC) -> THIS script carries the law into price
space by MC integration and VALIDATES it against the real series. The same
MC construction (integrate pure-noise steps, centered moving-average anchor,
ensemble statistics at the end) supplies the price-space observation
variance for the ssmix coupling estimator.

USAGE
-----
  python noise_movav_H.py                      # full plot
  python noise_movav_H.py --window 30 --N 500  # template defaults
  python noise_movav_H.py --laplace            # Laplace band instead of Student-t
  python noise_movav_H.py --log                # log x-axis

OPTIONS
  --window n   centered moving-average window for the anchor [days] (default 30)
  --N n        MC paths per anchor day (default 500)
  --horizon n  MC horizon per anchor [days] (default 1 = daily noise cone)
  --Q q        max band percentile (default 99)
  --laplace    use the Laplace law instead of Student-t
  --log        logarithmic x-axis

OUTPUT
------
- per-cycle fitted noise scales (stdout)
- noise_movav_H.png: real hashrate + Q1..Q99 MC noise band (cycle-coloured)

DEPENDENCIES
------------
numpy, pandas, scipy, matplotlib  (+ local modules data_io.py, dailynHR.py)
Needs cm_data.csv -- run  python fetch_coinmetrics.py  first.
