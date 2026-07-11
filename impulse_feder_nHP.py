#!/usr/bin/env python3
# "Feder" control run (Codex finding 1): add a small explicit L2 term lam2*I on
# top of the smoothness penalty lam*D'D. The L2 spring pulls the kernel LEVEL
# (and thus H(200), the permanent effect) toward zero - it works AGAINST the
# P->HR finding. Question: at which spring strength does H(200) lose its sign
# or significance? lam2 = rho * mean(diag(Xc'Xc)); fresh CV for lam per rho.
import os
import numpy as np, pandas as pd
from datetime import date
from scipy.linalg import cho_factor, cho_solve
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('/home/hz/Data/hz.mplstyle')
mpl.rcParams['font.sans-serif'] = ['Comfortaa', 'DejaVu Sans', 'Arial']
BASE = os.path.dirname(os.path.abspath(__file__))

GENESIS = date(2009,1,3); K = 200; D_ = 11; START = '2012-03-13'
N_FOLDS = 5; N_BOOT = 500; BLOCK = 180
LAMBDAS = np.logspace(-1, 14, 31)
RHOS = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
rng = np.random.default_rng(42)
pz = pd.read_csv('/home/hz/Data/ziel.csv', sep=' ', header=None, names=['d','p','date'])
pz['date'] = pd.to_datetime(pz['date'], format='%d.%m.%Y'); pz = pz.sort_values('date').set_index('date')['p']
hr = pd.read_csv('/home/hz/Data/zeitgeist/ziel_HR.csv', sep=' ', header=None, names=['h','p2','date'])
hr['date'] = pd.to_datetime(hr['date'], format='%Y.%m.%d'); hr = hr.sort_values('date').set_index('date')['h']
end = min(pz[pz>0].index.max(), hr[hr>0].index.max()); cal = pd.date_range(START, end, freq='D')
lp = np.log(pz.reindex(cal).where(lambda s: s>0)).interpolate(limit_area='inside').to_numpy()
lh = np.log(hr.reindex(cal).where(lambda s: s>0)).interpolate(limit_area='inside').to_numpy()
t = np.array([(d.date()-GENESIS).days for d in cal], float); lt = np.log(t)
D = np.zeros((K,K+1)); D[np.arange(K),np.arange(K)] = -1; D[np.arange(K),np.arange(K)+1] = 1
DtD = D.T @ D; I = np.eye(K+1)

def run(driver, target, dirn):
    rows = np.arange(K, len(driver))
    X = np.stack([driver[rows-k] for k in range(K+1)], axis=1); y = target[rows]
    xm_full = X.mean(0); Xc_full = X - xm_full
    gbar = np.mean(np.diag(Xc_full.T @ Xc_full))
    folds = np.array_split(np.arange(len(y)), N_FOLDS)
    # per-fold Grams once
    FG = []
    for va in folds:
        keep = np.ones(len(y), bool)
        keep[max(0,va[0]-(K+D_)):min(len(y),va[-1]+(K+D_)+1)] = False
        tr = np.where(keep)[0]
        xm, ym = X[tr].mean(0), y[tr].mean()
        Xc, yc = X[tr]-xm, y[tr]-ym
        FG.append((Xc.T@Xc, Xc.T@yc, xm, ym, va))
    out = []
    for rho in RHOS:
        print(f"computing {dirn} rho={rho:g} ...", flush=True)
        lam2 = rho * gbar
        mse = np.zeros(len(LAMBDAS))
        for G, g, xm, ym, va in FG:
            for j, lam in enumerate(LAMBDAS):
                h = np.linalg.solve(G + lam*DtD + lam2*I + 1e-9*I, g)
                mse[j] += np.mean((y[va]-(X[va]@h+ym-xm@h))**2)
        lam = LAMBDAS[mse.argmin()]
        A = Xc_full.T@Xc_full + lam*DtD + lam2*I + 1e-9*I
        h = np.linalg.solve(A, Xc_full.T@(y-y.mean()))
        H = np.cumsum(h)
        pred = X@h + (y.mean()-xm_full@h); resid = y-pred
        cho = cho_factor(A); XcT = Xc_full.T
        nb = int(np.ceil(len(y)/BLOCK)); Hk = np.empty(N_BOOT)
        for i in range(N_BOOT):
            eps = rng.choice(np.array([-1.,1.]), nb).repeat(BLOCK)[:len(y)]
            hb = cho_solve(cho, XcT@((pred+eps*resid)-(pred+eps*resid).mean()))
            Hk[i] = np.cumsum(hb)[K]
        lo, hi = np.percentile(Hk, [5,95])
        sig = ' *' if lo>0 or hi<0 else '  '
        print(f"{dirn:6} rho={rho:7.3f}  lam2={lam2:9.2e}  lam*={lam:8.1e}  "
              f"H(200)={H[K]:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
        out.append((H[K], lo, hi))
    return out

dlt = lt[D_:]-lt[:-D_]
bP = (lp[D_:]-lp[:-D_])/dlt; bHR = (lh[D_:]-lh[:-D_])/dlt
res_ph = run(bP, bHR, 'P->HR')
print()
res_hp = run(bHR, bP, 'HR->P')
print("\nrho = lam2 / mean(diag(Xc'Xc)); rho=1 -> spring as strong as the data term")

# ------------------------------ plot: H(200) vs spring strength -----------
xs = np.arange(len(RHOS))
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_facecolor('#1a1a1a')
ax.axhline(0, color='gray', linewidth=0.8, alpha=0.6)
for res, col, lab in ((res_hp, 'cornflowerblue', r'$H^{HR\to P}(200)$'),
                      (res_ph, 'orange', r'$H^{P\to HR}(200)$')):
    ax.fill_between(xs, [r[1] for r in res], [r[2] for r in res],
                    color=col, alpha=0.15)
    ax.plot(xs, [r[0] for r in res], color=col, linewidth=2, marker='o', label=lab)
ax.set_xticks(xs)
ax.set_xticklabels(['0', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', '1'])
ax.set_xlabel(r'spring strength $\rho = \lambda_2\,/\,\langle\mathrm{diag}(X^{T}X)\rangle$')
ax.set_ylabel(r'$H(200)$')
ax.legend(loc='upper right', fontsize=10, facecolor='#1A1A1A',
          edgecolor='#808080', labelcolor='#E0E0E0')

with plt.rc_context({'text.usetex': False}):
    plt.suptitle('L2 Spring Control - Permanent Effect vs Spring Strength',
                 color='#CCCCCC', fontsize=14, y=0.985,
                 fontname='Comfortaa', fontweight='bold')
plt.figtext(0.5, 0.935, r'penalty $\lambda D^{T}D + \lambda_2 I$: the spring pulls '
            r'the kernel level toward zero, i.e. AGAINST a permanent effect --- '
            r'$\delta$=11d causal exponents, fresh CV per $\rho$, block-wild '
            'bootstrap 5--95\% on $H(200)$',
            ha='center', va='top', color='#999999', fontsize=10)
plt.subplots_adjust(top=0.90, bottom=0.09, left=0.08, right=0.92)
plt.savefig(os.path.join(BASE, 'impulse_feder_nHP.png'), dpi=300, facecolor='#0a0a0a')

plt.show()
