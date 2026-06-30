scale_inv_HP.py -- Scale-invariance test for price and hashrate
===============================================================

IDEA
----
A pure power law  X ~ t^n  is SCALE INVARIANT: the ratio X_2/X_1 depends only
on the ratio t_2/t_1, not on where in time you look. This test checks that
directly, symmetrically, with random unordered time pairs, for the BTC price
and the hashrate side by side.

MATH / METHOD
-------------
Draw N random UNORDERED index pairs (t_1, t_2) -- t_2 need NOT be later than
t_1, so log(t_2/t_1) can be negative. For each pair plot

    y = log(X_2 / X_1)   vs   x = log(t_2 / t_1).

If X ~ t^n is scale-invariant, the cloud COLLAPSES onto a straight line through
the origin with slope n. Curvature / an S-shape means the exponent itself
changes with scale (the relation is NOT a single clean power law). The slope is
fit by least squares THROUGH THE ORIGIN (fit_intercept=False), since a true
scale-invariant law forces the line through (0, 0).

Three panels:
  1 -- price (gold) and hashrate (blue), full scale, with the fitted slope line
  2 -- price pairs coloured by halving cycle (both endpoints same cycle), zoomed
  3 -- hashrate pairs coloured by halving cycle, zoomed (same axes as panel 2)
What to look for: a tight straight line through the origin = scale-invariant;
curvature or an S-shape in the tails = a scale-dependent exponent.

USAGE
-----
  python scale_inv_HP.py             # 100000 random pairs (default)
  python scale_inv_HP.py --n 50000   # fewer pairs (faster)
  python scale_inv_HP.py --no-plot   # print fitted slope only

OPTIONS
  --n N       number of random pairs (default 100000)
  --no-plot   compute and print only, no figure

OUTPUT
------
- fitted price exponent (slope) and same-cycle pair counts (stdout)
- scale_inv_HP.png: the three-panel scatter described above

DEPENDENCIES
------------
numpy, matplotlib, scikit-learn  (+ local modules data_io.py, dailynHR.py)
Needs cm_data.csv -- run  python fetch_coinmetrics.py  first.
