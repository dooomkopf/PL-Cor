dailyn_HP_evol.py -- Hashrate vs Price power-law exponent drift (overlay)
=========================================================================

IDEA
----
See dailyn_P_evol_README.txt for the concept and the math. This script only adds:
it shows the slow drift of the SG-smoothed (averaged) daily exponents of both the
hashrate and the price, overlaid in one panel.

MATH / METHOD
-------------
Local (daily) power-law exponent between two consecutive days:

    n(t) = log(X_2 / X_1) / log(t_2 / t_1)        (see dailynHR.daily_exponent)

with t = days since the genesis block. The raw daily exponent is extremely
noisy, so a Savitzky-Golay filter (local polynomial of order POLYORDER) of
window SG_WIN extracts the slow drift. The +/- win/2 edges are cropped because
the SG polynomial extrapolates there.

Only the long (default 365-day) SG curve is shown -- the model-consistent view
for comparing the two slow drifts.

USAGE
-----
  python dailyn_HP_evol.py                 # SG-365 for both, default
  python dailyn_HP_evol.py --SG-H 181 --SG-P 365   # separate windows
  python dailyn_HP_evol.py --no-plot       # print drift ranges only

OPTIONS
  --SG-H N   Savitzky-Golay window for the HASHRATE exponent [days]
  --SG-P N   Savitzky-Golay window for the PRICE exponent [days]
  --no-plot  compute and print only, no figure

OUTPUT
------
- drift range of n_HR and n_P (stdout)
- dailyn_HP_evol.png: the two SG-filtered exponent drifts, with halving guides

DEPENDENCIES
------------
numpy, scipy, matplotlib  (+ local modules data_io.py, dailynHR.py)
Needs cm_data.csv -- run  python fetch_coinmetrics.py  first.
