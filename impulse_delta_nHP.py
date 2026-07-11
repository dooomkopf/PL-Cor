#!/usr/bin/env python3
"""
impulse_delta_nHP.py - CAUSAL impulse-response between the Bitcoin PRICE
exponent and the HASHRATE exponent, delta variants 45 d and 91 d only.
Companion to impulse_nHP.py (SG-365) and impulse_grid_nHP.py (full grid).

Signals (CAUSAL, no future in any point):
  backward two-point exponent over span delta, assigned to the RIGHT edge t:
    b_delta(t) = [ln X(t) - ln X(t-delta)] / [ln t - ln(t-delta)],  delta in {45, 91}

Window-overlap rule (grey zone in the plot): at lag k the driver value was
measured over [t-k-delta, t-k] and the target over [t-delta, t]. For k < delta
the two windows OVERLAP - they contain the same days, so correlation there is
"for free" (common shock), not causality. Only k >= delta is a clean lead.

Estimator: ridge with first-difference smoothness penalty D'D (identical to
impulse_nHP.py), lambda by blocked 5-fold CV with purge gap K+delta.

Uncertainty: block-WILD bootstrap (CHANGED vs. impulse_nHP.py / grid script,
Codex finding 02.07.2026): residual blocks stay WHERE they are on the timeline
and only get their sign flipped by a Rademacher draw (+1/-1 per 180 d block).
The circular block bootstrap moved quiet early-years residuals into the noisy
late years and vice versa - b_delta noise grows with t (ln t - ln(t-delta)
shrinks), so the moved blocks produced bands of the wrong width. Sign-flipping
in place preserves the local noise level exactly.
"""

import os
import numpy as np
import pandas as pd
from datetime import date
from scipy.linalg import cho_factor, cho_solve
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('/home/hz/Data/hz.mplstyle')
mpl.rcParams['font.sans-serif'] = ['Comfortaa', 'DejaVu Sans', 'Arial']

rng = np.random.default_rng(42)
BASE = os.path.dirname(os.path.abspath(__file__))
GENESIS = date(2009, 1, 3)          # Bitcoin genesis block -> t = days since genesis

# ------------------------------ parameters -------------------------------
DELTAS   = [45, 91]                 # backward-exponent spans [days]
K        = 200                      # max kernel lag [days] (economic lags <= 100-200)
N_FOLDS  = 5
N_BOOT   = 200                      # bootstrap replicates
BLOCK    = 180                      # bootstrap block length [days]
LAMBDAS  = np.logspace(-1, 14, 31)
START    = '2012-03-13'             # ziel.csv price is continuously DAILY from here

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


def signals_delta(d):
    """Causal backward two-point exponent over span d, right-edge assignment."""
    dlt = logt[d:] - logt[:-d]
    return (logP[d:] - logP[:-d]) / dlt, (logHR[d:] - logHR[:-d]) / dlt

# ------------------------------ estimator --------------------------------
D = np.zeros((K, K + 1))
D[np.arange(K), np.arange(K)]     = -1.0
D[np.arange(K), np.arange(K) + 1] =  1.0
DtD = D.T @ D


def design(driver):
    """Lagged design matrix X[i,k] = driver(row_i - k), rows K..len-1."""
    rows = np.arange(K, len(driver))
    X = np.stack([driver[rows - k] for k in range(K + 1)], axis=1)
    return X, rows


def cv_lambda(X, y, purge):
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


def fit_kernel(X, y, lam):
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    A = Xc.T @ Xc + lam * DtD + 1e-9 * np.eye(X.shape[1])
    h = np.linalg.solve(A, Xc.T @ yc)
    return h, ym - xm @ h


def bootstrap_band_wild(X, y, h_hat, a_hat, lam):
    """Block-WILD bootstrap: residual blocks stay in place, only the sign of
    each 180 d block is flipped at random -> local noise level (which grows
    with t for b_delta) is preserved exactly."""
    n, K1 = X.shape
    pred = X @ h_hat + a_hat
    resid = y - pred
    Xc = X - X.mean(axis=0)
    cho = cho_factor(Xc.T @ Xc + lam * DtD + 1e-9 * np.eye(K1))
    XcT = Xc.T
    nb = int(np.ceil(n / BLOCK))
    hb = np.empty((N_BOOT, K1))
    for i in range(N_BOOT):
        eps = rng.choice(np.array([-1.0, 1.0]), nb).repeat(BLOCK)[:n]
        ystar = pred + eps * resid
        hb[i] = cho_solve(cho, XcT @ (ystar - ystar.mean()))
    Hb = np.cumsum(hb, axis=1)
    return (np.percentile(hb, [5, 95], axis=0),
            np.percentile(Hb, [5, 95], axis=0))


def estimate(driver, target, purge, direction):
    X, rows = design(driver)
    y = target[rows]
    lam = cv_lambda(X, y, purge)
    h, a = fit_kernel(X, y, lam)
    H = np.cumsum(h)
    r2 = 1 - np.var(y - (X @ h + a)) / np.var(y)
    (h_lo, h_hi), (H_lo, H_hi) = bootstrap_band_wild(X, y, h, a, lam)
    return dict(dir=direction, lam=lam, r2=r2, n=len(y),
                h=h, H=H, h_lo=h_lo, h_hi=h_hi, H_lo=H_lo, H_hi=H_hi,
                std_fac=np.std(driver) / np.std(target),
                edge=(lam == LAMBDAS[0]) or (lam == LAMBDAS[-1]))

# ------------------------------ run --------------------------------------
runs = []
for d in DELTAS:
    bP, bHR = signals_delta(d)
    for drv, tgt, dirn in ((bHR, bP, 'HR->P'), (bP, bHR, 'P->HR')):
        r = estimate(drv, tgt, K + d, dirn)
        r.update(delta=d, label=rf'$\delta$={d}d')
        runs.append(r)
        print(f"done delta-{d:2d} {dirn}", flush=True)

# ------------------------------ plot: 2x2 --------------------------------
SH = {'HR->P': {45: '#9dbdff', 91: '#3a66cc'},
      'P->HR': {45: '#ffcf80', 91: '#d97a00'}}
COLI = {'HR->P': 0, 'P->HR': 1}

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2)
ax = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]
for row in ax:
    for a in row:
        a.set_facecolor('#1a1a1a')
        a.axhline(0, color='gray', linewidth=0.8, alpha=0.6)
        # grey zones: window overlap k < delta -> no causal reading there
        a.axvspan(0, DELTAS[0], color='gray', alpha=0.10)
        a.axvspan(0, DELTAS[1], color='gray', alpha=0.10)

ks = np.arange(K + 1)
for r in runs:
    j = COLI[r['dir']]
    c = SH[r['dir']][r['delta']]
    ax[0][j].fill_between(ks, r['h_lo'], r['h_hi'], color=c, alpha=0.15)
    ax[0][j].plot(ks, r['h'], color=c, linewidth=2, label=r['label'])
    ax[1][j].fill_between(ks, r['H_lo'], r['H_hi'], color=c, alpha=0.15)
    ax[1][j].plot(ks, r['H'], color=c, linewidth=2, label=r['label'])

DIRTEX = {0: r'HR$\to$P', 1: r'P$\to$HR'}
for j in range(2):
    ax[0][j].set_title(rf'\textbf{{kernel $h_k$ --- {DIRTEX[j]}}}', fontsize=12, pad=4)
    ax[1][j].set_title(rf'\textbf{{step response $H(k)$ --- {DIRTEX[j]}}}',
                       fontsize=12, pad=4)
    ax[1][j].set_xlabel('lag k (days)')
    for i in range(2):
        ax[i][j].legend(loc='upper right', fontsize=10, facecolor='#1A1A1A',
                        edgecolor='#808080', labelcolor='#E0E0E0')
ax[0][0].set_ylabel(r'$h_k$')
ax[1][0].set_ylabel(r'$H(k)$')

with plt.rc_context({'text.usetex': False}):
    plt.suptitle('Hashrate <-> Price Impulse Response - Causal Delta Exponents',
                 color='#CCCCCC', fontsize=14, y=0.985,
                 fontname='Comfortaa', fontweight='bold')
plt.figtext(0.5, 0.935, r'backward $\delta$-exponents 45/91d (causal, right-edge) '
            r'--- grey zone: $k<\delta$, measurement windows overlap, no causal '
            'reading --- block-wild bootstrap 5--95\%',
            ha='center', va='top', color='#999999', fontsize=10)
plt.subplots_adjust(top=0.90, bottom=0.06, left=0.07, right=0.93,
                    hspace=0.35, wspace=0.18)
plt.savefig(os.path.join(BASE, 'impulse_delta_nHP.png'), dpi=300, facecolor='#0a0a0a')

# ------------------------------ console ----------------------------------
print(f"\n{'variant':9} {'dir':6} {'n':>5} {'lam*':>8} {'R2':>6}  "
      f"{'h0':>8} {'pk':>4}  "
      f"{'H(91)':>8} {'H(181)':>8} {'H(200)':>8} {'[5%':>8} {'95%]':>8} {'H_std':>7}")
for r in runs:
    h, H = r['h'], r['H']
    pk = int(np.abs(h).argmax())
    print(f"delta-{r['delta']:<3d} {r['dir']:6} {r['n']:5d} {r['lam']:8.1e} {r['r2']:6.2f}  "
          f"{h[0]:+8.4f} {pk:4d}  "
          f"{H[91]:+8.3f} {H[181]:+8.3f} {H[K]:+8.3f} "
          f"{r['H_lo'][K]:+8.3f} {r['H_hi'][K]:+8.3f} {H[K] * r['std_fac']:+7.3f}"
          + ('   <-- LAMBDA GRID EDGE!' if r['edge'] else ''))

print("\nREADING GUIDE:")
print(" - grey zone k < delta: driver and target windows share days (common shock),")
print("   correlation there is free of charge - only k >= delta is a clean lead.")
print(" - bands: block-wild bootstrap (sign flip in place) - heteroskedasticity-safe.")
print(" - the two directions have different units: compare shapes and H_std, not heights.")

plt.show()
