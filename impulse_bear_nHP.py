#!/usr/bin/env python3
"""
impulse_bear_nHP.py - BEAR-REGIME interaction impulse response between the
Bitcoin PRICE exponent and the HASHRATE exponent.
Companion to impulse_delta_nHP.py; question (user 02.07.2026): does the
hashrate give the lead when BTC sits below its power-law bottom band?

Regime definition (EXACTLY the price_osci method, fit_power_law_bottom):
  quantile regression of ln(P) on ln(days since genesis) over the FULL ziel.csv
  history -> bottom line P_q(t) = exp(a) * t^b;  I_bear(t) = 1 if P(t) < P_q(t).
  Main run q = 10% (523 bear days, ~6 separate periods incl. 2015, 2016-cluster,
  2022/23, 2023, 2026), sensitivity q = 5% (260 days, 3 periods).
  ** DECLARED LOOK-AHEAD: the quantile line is fitted on the full history -
  "cheap" is defined with hindsight. Descriptive regime analysis, not a
  tradable signal. (Expanding-window variant = possible v2.) **

Signals: causal backward delta-exponents (right edge), delta = 11 d main
(optimum of the 7..45 d sweep: stability plateau everywhere, pure-lead tail
significance holds only for delta <= ~13), delta = 25 d as control.

Model (interaction kernel, FULL sample - no bear-subset fit; three guards
against the selection traps discussed 02.07.2026):
  y(t) = alpha + c*I_bear(t)                     [bear level dummy   - trap A]
       + sum_{j=1..60}  a_j y(t-j)               [AR baseline        - trap A]
       + sum_{k=0..200} h_k x(t-k)               [base response]
       + sum_{k=0..200} g_k x(t-k) * I_bear(t)   [BEAR EXTRA kernel  = the test]
       + eps(t)
  Test statistic G(K) = sum_k g_k (+ tail version k > delta): does the driver
  push EXTRA when the system sits below the bottom band? g == 0 -> no bear
  difference. Both directions are fitted (hypothesis: HR->P; mirror P->HR).

Estimator: ridge with first-difference smoothness penalty PER BLOCK
(block-diagonal D'D on AR / h / g; dummy unpenalized), ONE shared lambda by
blocked 5-fold CV (contiguous folds, purge K+delta). Declared simplification:
separate lambdas per block would be cleaner (v2).
Uncertainty: block-wild bootstrap (sign flip per 180 d block, in place -
heteroskedasticity-safe). Validation: LEAVE-ONE-PERIOD-OUT - bear periods
(mask runs merged over gaps < 90 d) are dropped one at a time and G(K) is
refit at fixed lambda*: if the sign hangs on a single period, it shows here.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from datetime import date
from scipy.linalg import cho_factor, cho_solve, block_diag
import matplotlib.pyplot as plt

plt.style.use('/home/hz/Data/hz.mplstyle')

rng = np.random.default_rng(42)
BASE = os.path.dirname(os.path.abspath(__file__))
GENESIS = date(2009, 1, 3)          # Bitcoin genesis block -> t = days since genesis

# ------------------------------ parameters -------------------------------
DELTA_MAIN = 11                     # sweep optimum (7..45d: plateau; tail sig <= ~13)
DELTA_CTRL = 25                     # control span
QS         = (0.10, 0.05)           # bottom-quantile regimes: main 10%, sens. 5%
K          = 200                    # max kernel lag [days]
K_AR       = 60                     # AR-baseline lags (bear episodes are 30-73 d)
N_FOLDS    = 5
N_BOOT     = 500                    # bootstrap replicates
BLOCK      = 180                    # bootstrap block length [days]
LAMBDAS    = np.logspace(-1, 14, 31)
START      = '2012-03-13'           # ziel.csv price is continuously DAILY from here
GAP_MERGE  = 90                     # merge bear-mask runs closer than this [days]

# ------------------------------ data (as in impulse_delta_nHP.py) --------
def days_since_genesis(idx):
    return np.array([(d.date() - GENESIS).days for d in idx], dtype=float)

pz = pd.read_csv('/home/hz/Data/ziel.csv', sep=' ', header=None,
                 names=['days', 'price', 'date'])
pz['date'] = pd.to_datetime(pz['date'], format='%d.%m.%Y')
pz = pz.sort_values('date')

hr = pd.read_csv('/home/hz/Data/zeitgeist/ziel_HR.csv', sep=' ', header=None,
                 names=['hr', 'price2', 'date'])
hr['date'] = pd.to_datetime(hr['date'], format='%Y.%m.%d')
hr = hr.sort_values('date').set_index('date')['hr']

pser = pz.set_index('date')['price']
end = min(pser[pser > 0].index.max(), hr[hr > 0].index.max())
cal = pd.date_range(START, end, freq='D')
P_cal = pser.reindex(cal).where(lambda s: s > 0).interpolate(limit_area='inside')
logP  = np.log(P_cal).to_numpy()
logHR = np.log(hr.reindex(cal).where(lambda s: s > 0)
               .interpolate(limit_area='inside')).to_numpy()
assert np.isfinite(logP).all() and np.isfinite(logHR).all(), \
    "leading/trailing NaN in P/HR over [START, end] - check the window"
t    = days_since_genesis(cal)
logt = np.log(t)

# ------------------------------ bear regime (price_osci method) ----------
full = pz[pz['price'] > 0]                       # FULL history, raw rows
Xq = sm.add_constant(np.log(full['days'].to_numpy(float)))
lyq = np.log(full['price'].to_numpy(float))

def bear_mask(q):
    """I_bear on the daily calendar: P below the q-quantile PL bottom line
    (QuantReg ln P ~ ln t over the full history - declared look-ahead)."""
    r = QuantReg(lyq, Xq).fit(q=q)
    a, b = r.params[0], r.params[1]
    return (P_cal.to_numpy() < np.exp(a) * t**b), b

def bear_periods(mask):
    """Contiguous mask runs, merged over gaps < GAP_MERGE days."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    runs = list(zip(edges[::2], edges[1::2]))
    per = []
    for s, e in runs:
        if per and s - per[-1][1] < GAP_MERGE:
            per[-1] = (per[-1][0], e)
        else:
            per.append((s, e))
    return per

# ------------------------------ signals ----------------------------------
def signals_delta(d):
    """Causal backward two-point exponent, right-edge assignment."""
    dlt = logt[d:] - logt[:-d]
    return (logP[d:] - logP[:-d]) / dlt, (logHR[d:] - logHR[:-d]) / dlt

# ------------------------------ estimator --------------------------------
def diff_DtD(m):
    D = np.zeros((m - 1, m))
    D[np.arange(m - 1), np.arange(m - 1)] = -1.0
    D[np.arange(m - 1), np.arange(m - 1) + 1] = 1.0
    return D.T @ D

# penalty: [dummy 1] + [AR K_AR] + [h K+1] + [g K+1], smoothness per block
PEN = block_diag(np.zeros((1, 1)), diff_DtD(K_AR), diff_DtD(K + 1), diff_DtD(K + 1))
I_G = 1 + K_AR + (K + 1)            # first column of the g block
NCOL = 1 + K_AR + 2 * (K + 1)

def build_design(driver, target, bear_b):
    """Rows K..M-1. Columns: [I_bear | y-lags 1..K_AR | x-lags 0..K | x-lags*I_bear]."""
    rows = np.arange(K, len(target))
    Xh = np.stack([driver[rows - k] for k in range(K + 1)], axis=1)
    Xa = np.stack([target[rows - j] for j in range(1, K_AR + 1)], axis=1)
    Ib = bear_b[rows].astype(float)
    X = np.hstack([Ib[:, None], Xa, Xh, Xh * Ib[:, None]])
    return X, target[rows], rows, int(Ib.sum())

def solve_pen(Xc, yc, lam):
    A = Xc.T @ Xc + lam * PEN + 1e-9 * np.eye(NCOL)
    return np.linalg.solve(A, Xc.T @ yc)

def cv_lambda(X, y, purge):
    n = len(y)
    folds = np.array_split(np.arange(n), N_FOLDS)
    mse = np.zeros(len(LAMBDAS))
    for va in folds:
        keep = np.ones(n, dtype=bool)
        keep[max(0, va[0] - purge):min(n, va[-1] + purge + 1)] = False
        tr = np.where(keep)[0]
        xm, ym = X[tr].mean(axis=0), y[tr].mean()
        Xc, yc = X[tr] - xm, y[tr] - ym
        G, g = Xc.T @ Xc, Xc.T @ yc
        for j, lam in enumerate(LAMBDAS):
            w = np.linalg.solve(G + lam * PEN + 1e-9 * np.eye(NCOL), g)
            mse[j] += np.mean((y[va] - (X[va] @ w + ym - xm @ w)) ** 2)
    return LAMBDAS[mse.argmin()]

def estimate(driver, target, bear_b, d, direction, label):
    X, y, rows, n_bear = build_design(driver, target, bear_b)
    lam = cv_lambda(X, y, K + d)
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    w = solve_pen(Xc, yc, lam)
    a0 = ym - xm @ w
    pred = X @ w + a0
    resid = y - pred
    r2 = 1 - np.var(resid) / np.var(y)
    g = w[I_G:]
    h = w[I_G - (K + 1):I_G]
    G = np.cumsum(g)
    H = np.cumsum(h)
    # block-wild bootstrap
    cho = cho_factor(Xc.T @ Xc + lam * PEN + 1e-9 * np.eye(NCOL))
    XcT = Xc.T
    nb = int(np.ceil(len(y) / BLOCK))
    Gk = np.empty(N_BOOT); Gt = np.empty(N_BOOT); Hk = np.empty(N_BOOT)
    Gb = np.empty((N_BOOT, K + 1))
    for i in range(N_BOOT):
        eps = rng.choice(np.array([-1.0, 1.0]), nb).repeat(BLOCK)[:len(y)]
        ystar = pred + eps * resid
        wb = cho_solve(cho, XcT @ (ystar - ystar.mean()))
        gb = np.cumsum(wb[I_G:]); hb = np.cumsum(wb[I_G - (K + 1):I_G])
        Gb[i] = gb; Gk[i] = gb[K]; Gt[i] = gb[K] - gb[d]; Hk[i] = hb[K]
    return dict(dir=direction, label=label, d=d, lam=lam, r2=r2, n=len(y),
                n_bear=n_bear, rows=rows, X=X, y=y,
                h=h, H=H, g=g, G=G,
                G_lo=np.percentile(Gb, 5, axis=0), G_hi=np.percentile(Gb, 95, axis=0),
                GK=(G[K], *np.percentile(Gk, [5, 95])),
                GT=(G[K] - G[d], *np.percentile(Gt, [5, 95])),
                HK=(H[K], *np.percentile(Hk, [5, 95])),
                edge=(lam == LAMBDAS[0]) or (lam == LAMBDAS[-1]))

def loeo(r, periods, d):
    """Leave-one-period-out: drop rows whose day falls in the period, refit at
    fixed lambda*, report G(K). Row i of the design maps to calendar index
    d + K + i (right-edge b assignment)."""
    caldx = d + K + np.arange(len(r['y']))
    out = []
    for (s, e) in periods:
        keep = (caldx < s) | (caldx >= e)
        Xt, yt = r['X'][keep], r['y'][keep]
        xm, ym = Xt.mean(axis=0), yt.mean()
        w = solve_pen(Xt - xm, yt - ym, r['lam'])
        out.append((s, e, np.cumsum(w[I_G:])[K]))
    return out

# ------------------------------ run --------------------------------------
results = []
masks = {}
for q in QS:
    mask, bq = bear_mask(q)
    masks[q] = mask
    print(f"regime q={q*100:.0f}%: PL-bottom exponent b={bq:.4f}, "
          f"bear days on calendar: {mask.sum()}", flush=True)
for q in QS:
    for d in (DELTA_MAIN, DELTA_CTRL):
        bP, bHR = signals_delta(d)
        bear_b = masks[q][d:]                 # align mask to right-edge b grid
        for drv, tgt, dirn in ((bHR, bP, 'HR->P'), (bP, bHR, 'P->HR')):
            r = estimate(drv, tgt, bear_b, d, dirn, f'q={q*100:.0f}% d={d}')
            r.update(q=q)
            results.append(r)
            print(f"done {r['label']:12} {dirn}", flush=True)

main = [r for r in results if r['q'] == 0.10 and r['d'] == DELTA_MAIN]

# ------------------------------ plot: main combo (q=10%, delta=11) -------
COL = {'HR->P': 'cornflowerblue', 'P->HR': 'orange'}
ks = np.arange(K + 1)
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2)
ax = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]
for row in ax:
    for a in row:
        a.set_facecolor('#1a1a1a')
        a.axhline(0, color='gray', linewidth=0.8, alpha=0.6)
        a.axvspan(0, DELTA_MAIN, color='gray', alpha=0.12)

for j, r in enumerate(sorted(main, key=lambda r: r['dir'])):
    c = COL[r['dir']]
    ax[0][j].plot(ks, r['g'], color=c, linewidth=2, label=r'bear extra $g_k$')
    ax[0][j].plot(ks, r['h'], color='#888888', linewidth=1.2,
                  label=r'base $h_k$')
    ax[1][j].fill_between(ks, r['G_lo'], r['G_hi'], color=c, alpha=0.15)
    ax[1][j].plot(ks, r['G'], color=c, linewidth=2, label=r'bear extra $G(k)$')
    ax[1][j].plot(ks, r['H'], color='#888888', linewidth=1.2,
                  label=r'base $H(k)$')
    dirtex = r'HR$\to$P' if r['dir'] == 'HR->P' else r'P$\to$HR'
    ax[0][j].set_title(rf'\textbf{{kernels --- {dirtex}}}', fontsize=12, pad=4)
    ax[1][j].set_title(rf'\textbf{{step responses --- {dirtex}}}', fontsize=12, pad=4)
    ax[1][j].set_xlabel('lag k (days)')
    for i in range(2):
        ax[i][j].legend(loc='upper right', fontsize=10, facecolor='#1A1A1A',
                        edgecolor='#808080', labelcolor='#E0E0E0')
ax[0][0].set_ylabel(r'$h_k$, $g_k$')
ax[1][0].set_ylabel(r'$H(k)$, $G(k)$')

with plt.rc_context({'text.usetex': False}):
    plt.suptitle('Hashrate <-> Price Impulse Response - Bear-Regime Interaction',
                 color='#CCCCCC', fontsize=14, y=0.985,
                 fontweight='bold')
plt.figtext(0.5, 0.935, r'bear = price below the 10\% quantile PL line '
            '(full-history fit, declared look-ahead) --- '
            r'$\delta$=11d causal exponents, AR baseline + bear dummy, '
            'block-wild bootstrap 5--95\% on the bear-extra response $G$',
            ha='center', va='top', color='#999999', fontsize=10)
plt.subplots_adjust(top=0.90, bottom=0.06, left=0.07, right=0.93,
                    hspace=0.35, wspace=0.18)
plt.savefig(os.path.join(BASE, 'impulse_bear_nHP.png'), dpi=300, facecolor='#0a0a0a')

# ------------------------------ console ----------------------------------
print(f"\n{'variant':12} {'dir':6} {'n':>5} {'nBear':>5} {'lam*':>8} {'R2':>5}  "
      f"{'G(K)':>7} {'[5%':>7} {'95%]':>7}  {'Gtail':>7} {'[5%':>7} {'95%]':>7}  "
      f"{'H(K)':>7} {'[5%':>7} {'95%]':>7}")
for r in results:
    GK, GT, HK = r['GK'], r['GT'], r['HK']
    s1 = '*' if GK[1] > 0 or GK[2] < 0 else ' '
    s2 = '*' if GT[1] > 0 or GT[2] < 0 else ' '
    print(f"{r['label']:12} {r['dir']:6} {r['n']:5d} {r['n_bear']:5d} "
          f"{r['lam']:8.1e} {r['r2']:5.2f}  "
          f"{GK[0]:+7.3f} {GK[1]:+7.3f} {GK[2]:+7.3f}{s1} "
          f"{GT[0]:+7.3f} {GT[1]:+7.3f} {GT[2]:+7.3f}{s2} "
          f"{HK[0]:+7.3f} {HK[1]:+7.3f} {HK[2]:+7.3f}"
          + ('  <-- LAMBDA GRID EDGE!' if r['edge'] else ''))

print("\nLEAVE-ONE-PERIOD-OUT (main combo q=10%, delta=11, G(K) refit at lam*):")
periods = bear_periods(masks[0.10])
for r in main:
    parts = []
    for s, e, gk in loeo(r, periods, DELTA_MAIN):
        parts.append(f"{cal[s].strftime('%Y-%m')}:{gk:+.2f}")
    print(f"  {r['dir']:6} full={r['GK'][0]:+.2f} | without " + "  ".join(parts))

print("\nREADING GUIDE:")
print(" - G = EXTRA response while below the bottom band (on top of base h + AR")
print("   baseline + bear level dummy). G==0 -> bears are no different.")
print(" - grey zone k <= delta: overlapping measurement windows, no causal reading.")
print(" - look-ahead declared: the bottom line knows the full history.")
print(" - LOEO: if G's sign hangs on one single bear period, it shows above.")

plt.show()
