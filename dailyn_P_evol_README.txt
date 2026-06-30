dailyn_P_evol.py -- Price power-law exponent drift n_P(t)
=========================================================

IDEA
----
The distribution of the daily price exponent n_P (the actual noise of Bitcoin's
growth; see the math below) can be measured cleanly. On top of it sits a slow
drift, which the Savitzky-Golay filter makes visible while preserving the mean.
The drift is real: it stays stable across a wide range of filters and window
widths. The plot shows the raw daily exponent (one dot per day, coloured by
halving cycle) with SG curves of growing window length overlaid. Twin of
dailyn_H_evol.py (hashrate); same method, price.

MATH / METHOD
-------------
Local (daily) power-law exponent of  P ~ A * t^n  between consecutive days:

    n_P(t) = log(P_2 / P_1) / log(t_2 / t_1)       (see dailynHR.daily_exponent)

t = days since the genesis block. Three Savitzky-Golay filters (local
polynomial order POLYORDER) of windows SG_WINDOWS = [91, 181, 365] days reveal
the drift at increasing smoothness. SG edges (+/- win/2) are cropped (the SG
polynomial extrapolates there). Raw points are coloured per halving cycle.

USAGE
-----
  python dailyn_P_evol.py            # full plot
  python dailyn_P_evol.py --no-plot  # print drift stats only

OPTIONS
  --no-plot   compute and print only, no figure
(Window lengths, polynomial order and y-padding are in the parameter block.)

OUTPUT
------
- N, median, std of n_P and the SG drift ranges (stdout); SG-365 at each halving
- dailyn_P_evol.png: raw n_P (cycle-coloured) + SG-91/181/365 drift curves

DEPENDENCIES
------------
numpy, scipy, matplotlib  (+ local modules data_io.py, dailynHR.py)
Needs cm_data.csv -- run  python fetch_coinmetrics.py  first.
