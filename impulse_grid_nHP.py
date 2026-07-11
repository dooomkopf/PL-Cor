#!/usr/bin/env python3
"""
impulse_grid_nHP.py - ROBUSTNESS GRID for the impulse-response (Green's-function)
analysis between the Bitcoin PRICE exponent and the HASHRATE exponent.
Companion to impulse_nHP.py (single SG-365 variant) - that script stays untouched.

WHY A GRID (user, 02.07.2026): do not trust one filter. Scan two families:

  (a) SG grid    : centered Savitzky-Golay smoothing of the ln-series, windows
                   {45, 91, 181, 365} d, polyorder 2 (below 45 d only noise,
                   365 d is the maximum). Centered SG is ACAUSAL (zero-phase):
                   it mixes +-W/2 of future into every point. Local exponent
                   b = d ln X_sg / d ln t (forward difference, as in the other
                   zeitgeist scripts).
  (b) delta grid : backward two-point exponents - CAUSAL, no future in the point:
                     b_delta(t) = [ln X(t) - ln X(t-delta)] / [ln t - ln(t-delta)]
                   assigned to the RIGHT edge t; delta in {10, 30, 90} d.
                   Lead-lag structure becomes interpretable down to ~delta days.

Decision criterion is STABILITY, not "best response": a feature (sign of h0,
sign/level of H at fixed marks, direction asymmetry) that persists across all SG
windows AND the causal delta variants is market; a feature that moves with the
filter width is apparatus (cf. the k~28 knee = target-window length in the ETF
reference case, ETFs/etf_impulse.py).

Estimator per variant x direction - IDENTICAL to impulse_nHP.py:
  ridge with first-difference smoothness penalty D'D, free intercept via
  centering, lambda by blocked 5-fold CV (contiguous folds, purge both sides),
  uncertainty by block-residual bootstrap (block 180 d, 200 reps, 5-95% band).

Per-variant scaling:
  SG-W  : K = W + 180  (kernel must hold the +-W/2 smear AND ~6 months physics),
          purge = K + W (lagged-design overlap AND acausal SG raw support),
          edge drop W//2 on both sides (as in the other zeitgeist scripts).
  delta : K = 200      (causal, no smear; economic lags expected <= 100-200 d),
          purge = K + delta (lagged-design overlap AND backward-window support).

Known limitation (delta grid): the variance of raw b_delta grows with t (similar
Delta-ln-P noise divided by a shrinking Delta-ln-t) -> late years get more weight
in the LS fit; the block bootstrap absorbs this only partially. Read the delta
kernels as late-sample weighted.

Units: h^(HR->P) is "b_P units per b_HR unit" and h^(P->HR) the inverse - the raw
amplitudes of the two directions are NOT comparable. The console therefore also
prints H_std(K) = H(K) * sigma_driver / sigma_target, which puts both directions
on the same standardized scale.
"""

import os
import numpy as np
import pandas as pd
from datetime import date
from scipy.signal import savgol_filter
from scipy.linalg import cho_factor, cho_solve
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('/home/hz/Data/hz.mplstyle')
mpl.rcParams['font.sans-serif'] = ['Comfortaa', 'DejaVu Sans', 'Arial']

rng = np.random.default_rng(42)
BASE = os.path.dirname(os.path.abspath(__file__))
GENESIS = date(2009, 1, 3)          # Bitcoin genesis block -> t = days since genesis

# ------------------------------ parameters -------------------------------
SG_WINDOWS = [45, 91, 181, 365]     # user 02.07.2026: <45 only noise, max 365
SG_POLY    = 2                      # from the existing zeitgeist scripts
DELTAS     = [10, 30, 90]           # backward-exponent spans [days]
K_DELTA    = 200                    # user: economic lead-lags <= 100-200 d
N_FOLDS    = 5
N_BOOT     = 200                    # bootstrap replicates
BLOCK      = 180                    # bootstrap block length [days]
LAMBDAS    = np.logspace(-1, 14, 31)
START      = '2012-03-13'           # ziel.csv price is continuously DAILY from here

# ------------------------------ data (as in impulse_nHP.py) --------------
def days_since_genesis(idx):
    return np.array([(d.date() - GENESIS).days for d in idx], dtype=float)

# price: ziel.csv  (days price date, sep=' ', %d.%m.%Y)
pz = pd.read_csv('/home/hz/Data/ziel.csv', sep=' ', header=None,
                 names=['days', 'price', 'date'])
pz['date'] = pd.to_datetime(pz['date'], format='%d.%m.%Y')
pz = pz.sort_values('date').set_index('date')['price']

# hashrate: ziel_HR.csv  (hashrate price date, sep=' ', %Y.%m.%d) -> col 0 = hashrate
hr = pd.read_csv('/home/hz/Data/zeitgeist/ziel_HR.csv', sep=' ', header=None,
                 names=['hr', 'price2', 'date'])
hr['date'] = pd.to_datetime(hr['date'], format='%Y.%m.%d')
hr = hr.sort_values('date').set_index('date')['hr']

end = min(pz[pz > 0].index.max(), hr[hr > 0].index.max())
cal = pd.date_range(START, end, freq='D')
P, HRs = pz.reindex(cal), hr.reindex(cal)
logP  = np.log(P.where(P > 0)).replace([np.inf, -np.inf], np.nan)
logHR = np.log(HRs.where(HRs > 0)).replace([np.inf, -np.inf], np.nan)
logP  = logP.interpolate(method='linear', limit_area='inside').to_numpy()
logHR = logHR.interpolate(method='linear', limit_area='inside').to_numpy()
assert np.isfinite(logP).all() and np.isfinite(logHR).all(), \
    "leading/trailing NaN in P/HR over [START, end] - check the window"
t    = days_since_genesis(cal)
logt = np.log(t)

# ------------------------------ signal variants --------------------------
def signals_sg(win):
    """Acausal variant: SG-smooth the LOG series (window win, polyorder 2),
    forward-difference exponent b = dln(X_sg)/dln(t), drop win//2 edge days."""
    dlt  = np.diff(logt)
    bP   = np.diff(savgol_filter(logP,  win, SG_POLY)) / dlt
    bHR  = np.diff(savgol_filter(logHR, win, SG_POLY)) / dlt
    half = win // 2
    sl   = slice(half, len(bP) - half)
    return bP[sl], bHR[sl]


def signals_delta(d):
    """Causal variant: backward two-point exponent over span d, assigned to the
    RIGHT edge -> no future enters the point."""
    dlt = logt[d:] - logt[:-d]
    return (logP[d:] - logP[:-d]) / dlt, (logHR[d:] - logHR[:-d]) / dlt

# ------------------------------ estimator (as in impulse_nHP.py) ---------
def make_DtD(K):
    D = np.zeros((K, K + 1))
    D[np.arange(K), np.arange(K)]     = -1.0
    D[np.arange(K), np.arange(K) + 1] =  1.0
    return D.T @ D


def design(driver, K):
    """Lagged design matrix X[i,k] = driver(row_i - k), rows K..len-1."""
    rows = np.arange(K, len(driver))
    X = np.stack([driver[rows - k] for k in range(K + 1)], axis=1)
    return X, rows


def cv_lambda(X, y, DtD, purge):
    """Blocked CV: 5 contiguous folds, purge gap on both sides. Gram matrix is
    lambda-independent -> computed once per fold, then only re-solved."""
    n, K1 = X.shape
    folds = np.array_split(np.arange(n), N_FOLDS)
    mse = np.zeros(len(LAMBDAS))
    eye = 1e-9 * np.eye(K1)
    for va in folds:
        keep = np.ones(n, dtype=bool)
        keep[max(0, va[0] - purge):min(n, va[-1] + purge + 1)] = False
        tr = np.where(keep)[0]
        xm, ym = X[tr].mean(axis=0), y[tr].mean()
        Xc, yc = X[tr] - xm, y[tr] - ym
        G, g = Xc.T @ Xc, Xc.T @ yc
        for j, lam in enumerate(LAMBDAS):
            h = np.linalg.solve(G + lam * DtD + eye, g)
            mse[j] += np.mean((y[va] - (X[va] @ h + ym - xm @ h)) ** 2)
    return LAMBDAS[mse.argmin()]


def fit_kernel(X, y, lam, DtD):
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    A = Xc.T @ Xc + lam * DtD + 1e-9 * np.eye(X.shape[1])
    h = np.linalg.solve(A, Xc.T @ yc)
    return h, ym - xm @ h


def bootstrap_band(X, y, h_hat, a_hat, lam, DtD):
    """Block-residual bootstrap conditional on the observed driver path X."""
    n, K1 = X.shape
    resid = y - (X @ h_hat + a_hat)
    Xc = X - X.mean(axis=0)
    cho = cho_factor(Xc.T @ Xc + lam * DtD + 1e-9 * np.eye(K1))
    XcT = Xc.T
    nb = int(np.ceil(n / BLOCK))
    hb = np.empty((N_BOOT, K1))
    for i in range(N_BOOT):
        starts = rng.integers(0, n, nb)
        idx = np.concatenate([(s + np.arange(BLOCK)) % n for s in starts])[:n]
        ystar = (X @ h_hat + a_hat) + resid[idx]
        hb[i] = cho_solve(cho, XcT @ (ystar - ystar.mean()))
    Hb = np.cumsum(hb, axis=1)
    return (np.percentile(hb, [5, 95], axis=0),
            np.percentile(Hb, [5, 95], axis=0))


def estimate(driver, target, K, purge, direction):
    X, rows = design(driver, K)
    y = target[rows]
    DtD = make_DtD(K)
    lam = cv_lambda(X, y, DtD, purge)
    h, a = fit_kernel(X, y, lam, DtD)
    H = np.cumsum(h)
    r2 = 1 - np.var(y - (X @ h + a)) / np.var(y)
    (h_lo, h_hi), (H_lo, H_hi) = bootstrap_band(X, y, h, a, lam, DtD)
    return dict(dir=direction, K=K, lam=lam, r2=r2, n=len(y),
                h=h, H=H, h_lo=h_lo, h_hi=h_hi, H_lo=H_lo, H_hi=H_hi,
                std_fac=np.std(driver) / np.std(target),
                edge=(lam == LAMBDAS[0]) or (lam == LAMBDAS[-1]))

# ------------------------------ run the grid -----------------------------
runs = []
for W in SG_WINDOWS:
    bP, bHR = signals_sg(W)
    K = W + 180
    for drv, tgt, dirn in ((bHR, bP, 'HR->P'), (bP, bHR, 'P->HR')):
        r = estimate(drv, tgt, K, K + W, dirn)
        r.update(grid='SG', label=f'SG-{W}')
        runs.append(r)
        print(f"done {r['label']:9} {dirn}", flush=True)
for d in DELTAS:
    bP, bHR = signals_delta(d)
    for drv, tgt, dirn in ((bHR, bP, 'HR->P'), (bP, bHR, 'P->HR')):
        r = estimate(drv, tgt, K_DELTA, K_DELTA + d, dirn)
        r.update(grid='delta', label=f'delta-{d}')
        runs.append(r)
        print(f"done {r['label']:9} {dirn}", flush=True)

# ------------------------------ plot: 4x2 grid ---------------------------
SH_SG = {'HR->P': ['#d0e0ff', '#9dbdff', '#6b95f5', '#3a66cc'],
         'P->HR': ['#ffe9c2', '#ffcf80', '#ff9d33', '#d97a00']}
SH_DL = {'HR->P': ['#9dbdff', '#6b95f5', '#3a66cc'],
         'P->HR': ['#ffcf80', '#ff9d33', '#d97a00']}
ROW0 = {'SG': 0, 'delta': 2}
COLI = {'HR->P': 0, 'P->HR': 1}
IDX  = {'SG':    {f'SG-{w}': i for i, w in enumerate(SG_WINDOWS)},
        'delta': {f'delta-{d}': i for i, d in enumerate(DELTAS)}}
LEG  = {'SG':    {f'SG-{w}': f'SG-{w}' for w in SG_WINDOWS},
        'delta': {f'delta-{d}': rf'$\delta$={d}d' for d in DELTAS}}

fig = plt.figure(figsize=(14, 18))
gs = fig.add_gridspec(4, 2)
ax = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(4)]
for row in ax:
    for a in row:
        a.set_facecolor('#1a1a1a')
        a.axhline(0, color='gray', linewidth=0.8, alpha=0.6)

for r in runs:
    i0, j = ROW0[r['grid']], COLI[r['dir']]
    sh = (SH_SG if r['grid'] == 'SG' else SH_DL)[r['dir']]
    c  = sh[IDX[r['grid']][r['label']]]
    ks = np.arange(r['K'] + 1)
    lab = LEG[r['grid']][r['label']]
    ax[i0][j].plot(ks, r['h'], color=c, linewidth=1.8, label=lab)
    ax[i0 + 1][j].fill_between(ks, r['H_lo'], r['H_hi'], color=c, alpha=0.13)
    ax[i0 + 1][j].plot(ks, r['H'], color=c, linewidth=1.8, label=lab)

DIRTEX = {0: r'HR$\to$P', 1: r'P$\to$HR'}
for j in range(2):
    ax[0][j].set_title(rf'\textbf{{SG grid (acausal): kernel $h_k$ --- {DIRTEX[j]}}}',
                       fontsize=12, pad=4)
    ax[1][j].set_title(rf'\textbf{{SG grid: step response $H(k)$ --- {DIRTEX[j]}}}',
                       fontsize=12, pad=4)
    ax[2][j].set_title(rf'\textbf{{$\delta$ grid (causal): kernel $h_k$ --- {DIRTEX[j]}}}',
                       fontsize=12, pad=4)
    ax[3][j].set_title(rf'\textbf{{$\delta$ grid: step response $H(k)$ --- {DIRTEX[j]}}}',
                       fontsize=12, pad=4)
    ax[1][j].set_xlabel('lag k (days)')
    ax[3][j].set_xlabel('lag k (days)')
for i, lb in enumerate([r'$h_k$', r'$H(k)$', r'$h_k$', r'$H(k)$']):
    ax[i][0].set_ylabel(lb)
for i in (0, 2):
    for j in range(2):
        ax[i][j].legend(loc='upper right', fontsize=10, facecolor='#1A1A1A',
                        edgecolor='#808080', labelcolor='#E0E0E0')

with plt.rc_context({'text.usetex': False}):
    plt.suptitle('Hashrate <-> Price Impulse Response - Robustness Grid',
                 color='#CCCCCC', fontsize=14, y=0.985,
                 fontname='Comfortaa', fontweight='bold')
plt.figtext(0.5, 0.935, r'SG windows 45/91/181/365d (zero-phase, acausal) vs. '
            r'backward $\delta$-exponents 10/30/90d (causal) --- ridge + smoothness, '
            'blocked CV, bootstrap 5--95\% band on $H$',
            ha='center', va='top', color='#999999', fontsize=10)
plt.subplots_adjust(top=0.90, bottom=0.045, left=0.07, right=0.93,
                    hspace=0.5, wspace=0.18)
plt.savefig(os.path.join(BASE, 'impulse_grid_nHP.png'), dpi=300, facecolor='#0a0a0a')

# ------------------------------ console ----------------------------------
print(f"\n{'variant':9} {'dir':6} {'n':>5} {'lam*':>8} {'R2':>6}  "
      f"{'h0':>8} {'[5%':>8} {'95%]':>8} {'pk':>4}  "
      f"{'H(91)':>8} {'H(181)':>8} {'H(K)':>8} {'[5%':>8} {'95%]':>8} {'H_std(K)':>9}")
for r in runs:
    h, H, K = r['h'], r['H'], r['K']
    pk = int(np.abs(h).argmax())
    print(f"{r['label']:9} {r['dir']:6} {r['n']:5d} {r['lam']:8.1e} {r['r2']:6.2f}  "
          f"{h[0]:+8.4f} {r['h_lo'][0]:+8.4f} {r['h_hi'][0]:+8.4f} {pk:4d}  "
          f"{H[91]:+8.3f} {H[181]:+8.3f} {H[K]:+8.3f} "
          f"{r['H_lo'][K]:+8.3f} {r['H_hi'][K]:+8.3f} {H[K] * r['std_fac']:+9.3f}"
          + ('   <-- LAMBDA GRID EDGE!' if r['edge'] else ''))

print("\nREADING GUIDE:")
print(" - criterion is STABILITY: a sign/level that persists across SG-45..365 AND")
print("   the causal delta variants is market; what scales with the window is filter.")
print(" - SG variants are ACAUSAL (zero-phase): lag structure below ~W/2 is no causal timing.")
print(" - delta variants are CAUSAL: lead-lag readable down to ~delta days.")
print(" - the two directions have different units: compare shapes and H_std(K), not raw heights.")

plt.show()
