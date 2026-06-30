dailyn_H_evol.py -- Hashrate power-law exponent drift n_HR(t)
=============================================================

IDEA
----
See dailyn_P_evol_README.txt for the concept and the math. We carry over the
idea proven for the price: to view the signal in the space of the (assumed)
exponents. If the hashrate also grows as a power law, this move is legitimate --
and it additionally lets us analyse the noise in a space where signal and noise
can hopefully be separated more cleanly.

MATH / METHOD
-------------
Local (daily) power-law exponent of  H ~ A * t^n  between consecutive days:

    n_HR(t) = log(H_2 / H_1) / log(t_2 / t_1)      (see dailynHR.daily_exponent)

t = days since the genesis block. Three Savitzky-Golay filters (local
polynomial order POLYORDER) of windows SG_WINDOWS = [91, 181, 365] days reveal
the drift at increasing smoothness. SG edges (+/- win/2) are cropped (the SG
polynomial extrapolates there). The y-range is fixed and symmetric (YLIM) so
the large early swings stay comparable across runs.

USAGE
-----
  python dailyn_H_evol.py            # full plot
  python dailyn_H_evol.py --no-plot  # print drift stats only

OPTIONS
  --no-plot   compute and print only, no figure
(Window lengths, polynomial order and y-limits are in the parameter block.)

OUTPUT
------
- N, median, std of n_HR and the SG drift ranges (stdout); SG-365 at each halving
- dailyn_H_evol.png: raw n_HR (cycle-coloured) + SG-91/181/365 drift curves

DEPENDENCIES
------------
numpy, scipy, matplotlib  (+ local modules data_io.py, dailynHR.py)
Needs cm_data.csv -- run  python fetch_coinmetrics.py  first.
