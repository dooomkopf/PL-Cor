#!/usr/bin/env python3
"""
impulse_synth_nHP.py - PLANTED-TRUTH benchmark (user 04.07.2026, inspired by
lambda_demo.py): we cannot know the market's true kernel h - but we can test
whether OUR pipeline would recover a kernel LIKE ours under OUR conditions.

Differences to the didactic lambda_demo.py (which used a white driver, iid
noise and randomly permuted CV folds - all three forbidden here):
  - driver  = the REAL b_P series (delta=11): true autocorrelation/spectrum;
  - noise   = the REAL residuals of the delta=11 P->HR fit, re-randomised per
              replicate by block-wild sign flips (180 d) -> heteroskedastic
              variance ladder preserved;
  - pipeline = IDENTICAL to impulse_delta_nHP.py: contiguous 5-fold CV with
              purge K+delta, lambda grid logspace(-1,14,31), block-wild
              bootstrap band (200 reps).

Three planted truths (K=200):
  A 'persistent' : h_A(k) ~ exp(-k/60), scaled to H_true(200)=+0.28
                   (rise-and-stay - our P->HR finding);
  B 'transient'  : H_B(k) ~ k*exp(-k/50), scaled to peak +0.50, H(200)->~0.09
                   (hump-and-give-back - HR->P-like);
  C 'null'       : h == 0 (false-positive check).

Per truth, R Monte-Carlo replicates: y* = alpha + X h_true + noise*, full
pipeline on each. Reported: mean lambda*, bias of H(200), band coverage
(target ~90%), and for the null the false-positive rate (target ~10%).
"""

import os
import numpy as np
import pandas as pd
from datetime import date
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt

plt.style.use('/home/hz/Data/hz.mplstyle')

rng = np.random.default_rng(42)
BASE = os.path.dirname(os.path.abspath(__file__))
GENESIS = date(2009, 1, 3)

# ------------------------------ parameters -------------------------------
DELTA    = 11                       # working probe [days]
K        = 200                      # max kernel lag [days]
R        = 50                       # Monte-Carlo replicates per planted truth
N_FOLDS  = 5
N_BOOT   = 200
BLOCK    = 180
LAMBDAS  = np.logspace(-1, 14, 31)
LAM_REAL = 1e10                     # lambda* of the real delta=11 P->HR fit
START    = '2012-03-13'

# ------------------------------ data (as in impulse_delta_nHP.py) --------
pz = pd.read_csv('/home/hz/Data/ziel.csv', sep=' ', header=None,
                 names=['days', 'price', 'date'])
pz['date'] = pd.to_datetime(pz['date'], format='%d.%m.%Y')
pz = pz.sort_values('date').set_index('date')['price']
hr = pd.read_csv('/home/hz/Data/zeitgeist/ziel_HR.csv', sep=' ', header=None,
                 names=['hr', 'price2', 'date'])
hr['date'] = pd.to_datetime(hr['date'], format='%Y.%m.%d')
hr = hr.sort_values('date').set_index('date')['hr']
end = min(pz[pz > 0].index.max(), hr[hr > 0].index.max())
cal = pd.date_range(START, end, freq='D')
logP  = np.log(pz.reindex(cal).where(lambda s: s > 0)
               .interpolate(limit_area='inside')).to_numpy()
logHR = np.log(hr.reindex(cal).where(lambda s: s > 0)
               .interpolate(limit_area='inside')).to_numpy()
t    = np.array([(d.date() - GENESIS).days for d in cal], dtype=float)
logt = np.log(t)

dlt  = logt[DELTA:] - logt[:-DELTA]
b_P  = (logP[DELTA:]  - logP[:-DELTA])  / dlt      # REAL driver
b_HR = (logHR[DELTA:] - logHR[:-DELTA]) / dlt      # real target (residual source)

rows = np.arange(K, len(b_P))
X = np.stack([b_P[rows - k] for k in range(K + 1)], axis=1)
y_real = b_HR[rows]
n = len(y_real)

D = np.zeros((K, K + 1))
D[np.arange(K), np.arange(K)]     = -1.0
D[np.arange(K), np.arange(K) + 1] =  1.0
DtD = D.T @ D
EYE = 1e-9 * np.eye(K + 1)

# real fit at LAM_REAL -> residuals = the noise source of the benchmark
xm, ym = X.mean(axis=0), y_real.mean()
Xc = X - xm
A_real = Xc.T @ Xc + LAM_REAL * DtD + EYE
h_real = np.linalg.solve(A_real, Xc.T @ (y_real - ym))
alpha_real = ym - xm @ h_real
resid_real = y_real - (X @ h_real + alpha_real)
print(f"real fit: alpha={alpha_real:+.4f}, resid std={resid_real.std():.3f}, n={n}")

# ------------------------------ planted truths ---------------------------
kk = np.arange(K + 1)
hA = np.exp(-kk / 60.0);  hA *= 0.28 / hA.sum()          # H(200)=+0.28
HB = kk * np.exp(-kk / 50.0)
hB = np.diff(np.r_[0.0, HB]);  hB *= 0.50 / np.max(np.cumsum(hB))
hC = np.zeros(K + 1)
TRUTHS = [('A persistent', hA), ('B transient', hB), ('C null', hC)]

# ------------------------------ pipeline pieces --------------------------
folds = list(np.array_split(np.arange(n), N_FOLDS))
PURGE = K + DELTA
FOLD_PRE = []                       # per-fold: (tr_idx, Xc_tr, G_tr, xm_tr)
for va in folds:
    keep = np.ones(n, dtype=bool)
    keep[max(0, va[0] - PURGE):min(n, va[-1] + PURGE + 1)] = False
    tr = np.where(keep)[0]
    xmf = X[tr].mean(axis=0)
    Xcf = X[tr] - xmf
    FOLD_PRE.append((tr, Xcf, Xcf.T @ Xcf, xmf, va))

nb = int(np.ceil(n / BLOCK))

def pipeline(y):
    """Identical estimator: blocked+purged CV -> lambda*, fit, wild-boot band."""
    mse = np.zeros(len(LAMBDAS))
    for tr, Xcf, Gf, xmf, va in FOLD_PRE:
        ytr = y[tr]
        gf = Xcf.T @ (ytr - ytr.mean())
        for j, lam in enumerate(LAMBDAS):
            h = np.linalg.solve(Gf + lam * DtD + EYE, gf)
            mse[j] += np.mean((y[va] - (X[va] @ h + ytr.mean() - xmf @ h)) ** 2)
    lam = LAMBDAS[mse.argmin()]
    ymu = y.mean()
    A = Xc.T @ Xc + lam * DtD + EYE
    h = np.linalg.solve(A, Xc.T @ (y - ymu))
    a = ymu - xm @ h
    pred = X @ h + a
    resid = y - pred
    cho = cho_factor(A); XcT = Xc.T
    H200s = np.empty(N_BOOT)
    for i in range(N_BOOT):
        eps = rng.choice(np.array([-1.0, 1.0]), nb).repeat(BLOCK)[:n]
        ystar = pred + eps * resid
        hb = cho_solve(cho, XcT @ (ystar - ystar.mean()))
        H200s[i] = np.cumsum(hb)[K]
    lo, hi = np.percentile(H200s, [5, 95])
    return lam, np.cumsum(h), lo, hi

# ------------------------------ Monte-Carlo ------------------------------
results = {}
for name, h_true in TRUTHS:
    H_true = float(np.cumsum(h_true)[K])
    lams, H200, los, his, curves = [], [], [], [], []
    for r in range(R):
        eps = rng.choice(np.array([-1.0, 1.0]), nb).repeat(BLOCK)[:n]
        y = alpha_real + X @ h_true + eps * resid_real
        lam, Hc, lo, hi = pipeline(y)
        lams.append(lam); H200.append(Hc[K]); los.append(lo); his.append(hi)
        if len(curves) < 8:
            curves.append(Hc)
    lams, H200 = np.array(lams), np.array(H200)
    los, his = np.array(los), np.array(his)
    cover = np.mean((los <= H_true) & (H_true <= his))
    exclude0 = np.mean((los > 0) | (his < 0))
    results[name] = dict(h_true=h_true, H_true=H_true, lams=lams, H200=H200,
                         cover=cover, excl0=exclude0, curves=curves)
    print(f"{name:13s}: H_true={H_true:+.3f}  <H^>={H200.mean():+.3f}"
          f"+-{H200.std():.3f}  Bias={H200.mean()-H_true:+.3f}  "
          f"Coverage={cover:.0%}  Band-ohne-0={exclude0:.0%}  "
          f"lam* median={np.median(lams):.1e}", flush=True)

# ------------------------------ plot -------------------------------------
COLS = {'A persistent': 'orange', 'B transient': 'cornflowerblue',
        'C null': '#ba68c8'}
fig = plt.figure(figsize=(15, 6.2))
gs = fig.add_gridspec(1, 3)
for i, (name, h_true) in enumerate(TRUTHS):
    ax = fig.add_subplot(gs[0, i])
    r = results[name]; c = COLS[name]
    ax.set_facecolor('#1a1a1a')
    ax.axhline(0, color='gray', linewidth=0.8, alpha=0.6)
    for Hc in r['curves']:
        ax.plot(kk, Hc, color=c, linewidth=0.9, alpha=0.55)
    ax.plot(kk, np.cumsum(h_true), color='#e0e0e0', linewidth=2.2,
            linestyle='--', label=r'wahres $H(k)$ (gepflanzt)')
    ax.plot([], [], color=c, linewidth=0.9,
            label=rf'$\hat H(k)$, {len(r["curves"])} von {R} Replikaten')
    ax.set_title(rf'\textbf{{{name}}}', fontsize=12, pad=4)
    ax.set_xlabel('lag k (days)')
    if i == 0:
        ax.set_ylabel(r'$H(k)$')
    ax.legend(loc='lower right', fontsize=9, facecolor='#1A1A1A',
              edgecolor='#808080', labelcolor='#E0E0E0')
    ax.text(0.03, 0.96,
            rf'Bias $={r["H200"].mean() - r["H_true"]:+.3f}$'
            + '\n' + rf'Coverage $={r["cover"]:.0%}$'.replace('%', r'\%')
            + '\n' + rf'Band ohne 0: ${r["excl0"]:.0%}$'.replace('%', r'\%'),
            transform=ax.transAxes, ha='left', va='top',
            fontsize=9, color='#cccccc')

with plt.rc_context({'text.usetex': False}):
    plt.suptitle('Planted-Truth Benchmark - Recovers the Pipeline a Known Kernel?',
                 color='#CCCCCC', fontsize=14, y=0.985,
                 fontweight='bold')
plt.figtext(0.5, 0.935, rf'echter $b_P$-Treiber ($\delta$=11d), Rauschen = '
            r'echte Residuen mit Block-Wild-Vorzeichen, identische Pipeline '
            rf'(geblockte CV mit Purge, Wild-Bootstrap) --- $R={R}$ Replikate '
            r'je Wahrheit --- Coverage-Ziel $\approx$90\,\%, '
            r'Null-Fall Band-ohne-0-Ziel $\approx$10\,\%',
            ha='center', va='top', color='#999999', fontsize=10)
plt.subplots_adjust(top=0.84, bottom=0.10, left=0.05, right=0.95, wspace=0.20)
plt.savefig(os.path.join(BASE, 'impulse_synth_nHP.png'), dpi=300,
            facecolor='#0a0a0a')

print("\nREADING GUIDE:")
print(" - A: kommt H(200)=+0.28 ohne Bias zurueck, Coverage ~90%? Dann ist der")
print("   reale P->HR-Befund instrumentell glaubwuerdig.")
print(" - B: wird die Buckel-Form geborgen und H(200)~0 nicht als permanent verkauft?")
print(" - C: Band-ohne-0 sollte ~10% sein (nominales Niveau) - Falsch-Positiv-Check.")

plt.show()
