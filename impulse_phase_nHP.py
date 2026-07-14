#!/usr/bin/env python3
"""
impulse_phase_nHP.py - HALVING-PHASE conditioned impulse response between the
Bitcoin PRICE exponent and the HASHRATE exponent.
Companion to impulse_delta_nHP.py (base kernels) and impulse_bear_nHP.py
(bear-regime interaction). Question: does the lead-lag coupling differ across
the phases of the halving cycle?

Phase grid (EXOGENOUS halving clock - unlike the bear script there is NO
fitted regime and hence NO look-ahead):
  halving days since genesis: 0 (genesis counted as the 0th halving),
  1425, 2744, 4146, 5586.  For calendar day t let d = t - last halving <= t:
    d in [  0,100) -> P3    (P3 DELIBERATELY wraps around the halving:
    d in [100,550) -> P1     sluggish accumulation phase)
    d in [550,950) -> P2
    d >= 950       -> P3
  The data window starts at ~t=1165 and falls into P3 of the genesis cycle
  automatically - no special rule. Cycle ID of a row = index of the last
  halving (0..4).

Signals: causal backward delta-exponents (right edge, loader and formula
identical to impulse_delta_nHP.py), delta = 11 d, K = 200. Design rows
K..len-1; design row i maps to calendar day t[delta+K+i].

PART 1 - GATEKEEPER (interaction model, full sample, both directions):
  y(t) = alpha + c1 I_P1(t) + c2 I_P2(t)            [level dummies, P3 = ref]
       + sum_{j=1..60}  a_j y(t-j)                  [AR baseline / inertia]
       + sum_{k=0..200} h_k  x(t-k)                 [base response]
       + sum_{k=0..200} g1_k x(t-k) I_P1(t)         [P1 extra kernel]
       + sum_{k=0..200} g2_k x(t-k) I_P2(t)         [P2 extra kernel]
       + eps(t)
  Ridge with first-difference smoothness penalty PER BLOCK (AR/h/g1/g2;
  dummies unpenalized), ONE shared lambda by blocked 5-time-block CV with
  purge gap K+delta (declared simplification as in impulse_bear_nHP.py).
  Block-wild bootstrap (Rademacher sign per 180 d block, in place, B=500)
  for G1(K), G2(K) and the tails G(K)-G(delta). LOEO: each halving cycle
  (only IDs with >= 100 rows) is dropped once completely, refit at fixed
  lambda*, G1(K)/G2(K) reported per omission (sign stability).

PART 2 - SEPARATE PHASE KERNELS (both directions):
  Per phase P in {P1,P2,P3}: target rows = rows whose calendar day lies in P,
  POOLED over all cycles. The lag windows run on the REAL calendar and MAY
  reach into the previous phase - this is CONDITIONING ON THE TARGET PHASE,
  not a cut of the input history. Estimator exactly as impulse_delta_nHP.py
  (h kernel + centering only, NO AR block): the unconditional phase response,
  comparable to the H(200) values of the main analysis.
  lambda by LOCO-CV: validation set = all rows of ONE cycle in the phase,
  training = the other cycles' rows in the same phase, where training rows
  whose calendar day falls in [a-(K+delta), b+(K+delta)] of any contiguous
  validation segment [a,b] are removed (seam quarantine: in practice this
  concerns the P3 seams at the halving where late P3 of the old cycle and
  early P3 of the new cycle touch; same-phase segments of different cycles
  are otherwise years apart). Cycles with < 100 rows in the phase are never
  used as validation. Criterion: sum of per-validation-cycle MSE.
  Uncertainty: block-wild bootstrap band (B=500), Rademacher sign per 180 d
  block WITHIN each contiguous calendar segment of the phase (blocks never
  cross segment borders; segment = maximal run of consecutive phase days).
"""

import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
from scipy.linalg import cho_factor, cho_solve, block_diag
import matplotlib.pyplot as plt

plt.style.use('/home/hz/Data/hz.mplstyle')

rng = np.random.default_rng(42)
BASE = os.path.dirname(os.path.abspath(__file__))
GENESIS = date(2009, 1, 3)          # Bitcoin genesis block -> t = days since genesis

# ------------------------------ parameters -------------------------------
DELTA    = 11                       # backward-exponent span [days] (sweep optimum)
K        = 200                      # max kernel lag [days]
K_AR     = 60                       # AR-baseline lags (part 1 only)
N_FOLDS  = 5
N_BOOT   = 500                      # bootstrap replicates
BLOCK    = 180                      # bootstrap block length [days]
LAMBDAS  = np.logspace(-1, 14, 31)
START    = '2012-03-13'             # ziel.csv price is continuously DAILY from here
HALVINGS = np.array([0.0, 1425.0, 2744.0, 4146.0, 5586.0])  # genesis = 0th halving
MIN_ROWS = 100                      # min rows for a cycle to serve as validation/LOEO

# ------------------------------ data (as in impulse_delta_nHP.py) --------
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

# ------------------------------ phase grid --------------------------------
def phase_cycle(tv):
    """Exogenous halving clock: phase code (1/2/3) and cycle ID per day t."""
    cyc = np.searchsorted(HALVINGS, tv, side='right') - 1
    d = tv - HALVINGS[cyc]
    ph = np.full(tv.shape, 3, dtype=int)
    ph[(d >= 100) & (d < 550)] = 1
    ph[(d >= 550) & (d < 950)] = 2
    return ph, cyc

# ------------------------------ penalties ---------------------------------
def diff_DtD(m):
    D = np.zeros((m - 1, m))
    D[np.arange(m - 1), np.arange(m - 1)] = -1.0
    D[np.arange(m - 1), np.arange(m - 1) + 1] = 1.0
    return D.T @ D

DtD = diff_DtD(K + 1)               # part-2 kernel penalty (as impulse_delta)

# part-1 penalty: [dummies 2] + [AR K_AR] + [h K+1] + [g1 K+1] + [g2 K+1]
PEN1 = block_diag(np.zeros((2, 2)), diff_DtD(K_AR),
                  diff_DtD(K + 1), diff_DtD(K + 1), diff_DtD(K + 1))
I_H  = 2 + K_AR                     # first column of the h block
I_G1 = I_H + (K + 1)                # first column of the g1 block
I_G2 = I_G1 + (K + 1)               # first column of the g2 block
NCOL = 2 + K_AR + 3 * (K + 1)

# ------------------------------ part 1: gatekeeper ------------------------
def solve_gate(Xc, yc, lam):
    A = Xc.T @ Xc + lam * PEN1 + 1e-9 * np.eye(NCOL)
    return np.linalg.solve(A, Xc.T @ yc)


def cv_lambda_gate(X, y, purge):
    """Blocked CV: 5 contiguous time blocks, purge gap K+delta on both sides."""
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
            w = np.linalg.solve(G + lam * PEN1 + 1e-9 * np.eye(NCOL), g)
            mse[j] += np.mean((y[va] - (X[va] @ w + ym - xm @ w)) ** 2)
    return LAMBDAS[mse.argmin()]


def estimate_gate(driver, target, I1, I2, direction):
    """Full-sample interaction fit. Columns:
    [I_P1 | I_P2 | y-lags 1..K_AR | x-lags 0..K | x-lags*I_P1 | x-lags*I_P2]."""
    rows = np.arange(K, len(target))
    Xh = np.stack([driver[rows - k] for k in range(K + 1)], axis=1)
    Xa = np.stack([target[rows - j] for j in range(1, K_AR + 1)], axis=1)
    X = np.hstack([I1[:, None], I2[:, None], Xa, Xh,
                   Xh * I1[:, None], Xh * I2[:, None]])
    y = target[rows]
    lam = cv_lambda_gate(X, y, K + DELTA)
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    w = solve_gate(Xc, yc, lam)
    a0 = ym - xm @ w
    pred = X @ w + a0
    resid = y - pred
    r2 = 1 - np.var(resid) / np.var(y)
    g1, g2 = w[I_G1:I_G2], w[I_G2:]
    G1, G2 = np.cumsum(g1), np.cumsum(g2)
    # block-wild bootstrap: sign flip per 180 d block, blocks stay in place
    cho = cho_factor(Xc.T @ Xc + lam * PEN1 + 1e-9 * np.eye(NCOL))
    XcT = Xc.T
    nb = int(np.ceil(len(y) / BLOCK))
    G1k = np.empty(N_BOOT); G1t = np.empty(N_BOOT)
    G2k = np.empty(N_BOOT); G2t = np.empty(N_BOOT)
    for i in range(N_BOOT):
        eps = rng.choice(np.array([-1.0, 1.0]), nb).repeat(BLOCK)[:len(y)]
        ystar = pred + eps * resid
        wb = cho_solve(cho, XcT @ (ystar - ystar.mean()))
        g1b, g2b = np.cumsum(wb[I_G1:I_G2]), np.cumsum(wb[I_G2:])
        G1k[i] = g1b[K]; G1t[i] = g1b[K] - g1b[DELTA]
        G2k[i] = g2b[K]; G2t[i] = g2b[K] - g2b[DELTA]
    return dict(dir=direction, lam=lam, r2=r2, n=len(y), X=X, y=y,
                G1=G1, G2=G2,
                G1K=(G1[K], *np.percentile(G1k, [5, 95])),
                G1T=(G1[K] - G1[DELTA], *np.percentile(G1t, [5, 95])),
                G2K=(G2[K], *np.percentile(G2k, [5, 95])),
                G2T=(G2[K] - G2[DELTA], *np.percentile(G2t, [5, 95])),
                edge=(lam == LAMBDAS[0]) or (lam == LAMBDAS[-1]))


def loeo_gate(r, cyc_rows):
    """Leave-one-cycle-out: drop all rows of one halving cycle (>= MIN_ROWS
    rows only), refit at fixed lambda*, report G1(K)/G2(K)."""
    out = []
    for c in np.unique(cyc_rows):
        if (cyc_rows == c).sum() < MIN_ROWS:
            continue
        keep = cyc_rows != c
        Xt, yt = r['X'][keep], r['y'][keep]
        xm, ym = Xt.mean(axis=0), yt.mean()
        w = solve_gate(Xt - xm, yt - ym, r['lam'])
        out.append((int(c), np.cumsum(w[I_G1:I_G2])[K], np.cumsum(w[I_G2:])[K]))
    return out

# ------------------------------ part 2: phase kernels ---------------------
def fit_kernel(X, y, lam):
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    A = Xc.T @ Xc + lam * DtD + 1e-9 * np.eye(X.shape[1])
    h = np.linalg.solve(A, Xc.T @ yc)
    return h, ym - xm @ h


def loco_lambda(X, y, cald, cyc):
    """Leave-one-cycle-out CV within one phase: validation = one cycle's rows,
    training = other cycles' rows minus the seam quarantine (training rows
    whose calendar day falls in [a-(K+DELTA), b+(K+DELTA)] of any contiguous
    validation segment [a,b] are dropped). Criterion: sum of per-cycle MSE."""
    val_cycles = [c for c in np.unique(cyc) if (cyc == c).sum() >= MIN_ROWS]
    mse = np.zeros(len(LAMBDAS))
    eye = 1e-9 * np.eye(K + 1)
    for c in val_cycles:
        va = cyc == c
        keep = ~va
        vd = cald[va]
        brk = np.where(np.diff(vd) > 1.5)[0]
        starts = np.r_[0, brk + 1]
        ends = np.r_[brk, len(vd) - 1]
        for s, e in zip(starts, ends):
            a, b = vd[s], vd[e]
            keep &= ~((cald >= a - (K + DELTA)) & (cald <= b + (K + DELTA)))
        tr, vai = np.where(keep)[0], np.where(va)[0]
        xm, ym = X[tr].mean(axis=0), y[tr].mean()
        Xc, yc = X[tr] - xm, y[tr] - ym
        G, g = Xc.T @ Xc, Xc.T @ yc
        for j, lam in enumerate(LAMBDAS):
            h = np.linalg.solve(G + lam * DtD + eye, g)
            mse[j] += np.mean((y[vai] - (X[vai] @ h + ym - xm @ h)) ** 2)
    return LAMBDAS[mse.argmin()], len(val_cycles)


def boot_band_seg(X, y, h_hat, a_hat, lam, seg_len):
    """Block-wild bootstrap band for H(k): Rademacher sign per 180 d block,
    Rademacher signs drawn for FIXED 180 d blocks within each contiguous
    calendar segment of the phase (residuals stay in place)
    (never across segment borders), residuals stay in place."""
    pred = X @ h_hat + a_hat
    resid = y - pred
    Xc = X - X.mean(axis=0)
    cho = cho_factor(Xc.T @ Xc + lam * DtD + 1e-9 * np.eye(K + 1))
    XcT = Xc.T
    nbs = [int(np.ceil(L / BLOCK)) for L in seg_len]
    hb = np.empty((N_BOOT, K + 1))
    for i in range(N_BOOT):
        eps = np.concatenate([rng.choice(np.array([-1.0, 1.0]), nb)
                              .repeat(BLOCK)[:L] for nb, L in zip(nbs, seg_len)])
        ystar = pred + eps * resid
        hb[i] = cho_solve(cho, XcT @ (ystar - ystar.mean()))
    Hb = np.cumsum(hb, axis=1)
    return np.percentile(Hb, [5, 95], axis=0)


def estimate_phase(driver, target, ridx, cald, cyc, direction, pname):
    """Phase-conditioned kernel fit: rows = design rows whose calendar day
    lies in the phase (pooled over cycles); lag windows run on the real
    calendar and may reach into the previous phase."""
    rows_sig = K + ridx                       # signal-array index of each row
    X = np.stack([driver[rows_sig - k] for k in range(K + 1)], axis=1)
    y = target[rows_sig]
    lam, nval = loco_lambda(X, y, cald, cyc)
    h, a0 = fit_kernel(X, y, lam)
    H = np.cumsum(h)
    # contiguous calendar segments of the phase (row indices are day-spaced)
    brk = np.where(np.diff(ridx) > 1)[0]
    seg_len = np.diff(np.r_[0, brk + 1, len(ridx)])
    H_lo, H_hi = boot_band_seg(X, y, h, a0, lam, seg_len)
    return dict(dir=direction, phase=pname, n=len(y), nseg=len(seg_len),
                nval=nval, lam=lam, h=h, H=H, H_lo=H_lo, H_hi=H_hi,
                Hstd=H[K] * np.std(X[:, 0]) / np.std(y),
                edge=(lam == LAMBDAS[0]) or (lam == LAMBDAS[-1]))

# ------------------------------ run ---------------------------------------
bP, bHR = signals_delta(DELTA)
n_design = len(bP) - K
cald_rows = t[DELTA + K + np.arange(n_design)]     # row i -> calendar day t[delta+K+i]
ph_rows, cyc_rows = phase_cycle(cald_rows)
I1 = (ph_rows == 1).astype(float)
I2 = (ph_rows == 2).astype(float)

print(f"Datenstand: letzter Kalendertag = {cal[-1].date()} "
      f"(t = {int(t[-1])} Tage seit Genesis), n_Design = {n_design}")
print(f"Phasen-Raster (exogene Halving-Uhr, d = Tage seit letztem Halving): "
      f"P3 [0,100)+[950,inf), P1 [100,550), P2 [550,950)")
print(f"Zeilen je Phase: P1 = {int(I1.sum())}, P2 = {int(I2.sum())}, "
      f"P3 = {int((ph_rows == 3).sum())}")
print(f"Zeilen je Zyklus: " + ", ".join(
    f"C{c}({(GENESIS + timedelta(days=int(HALVINGS[c]))).year}) = "
    f"{int((cyc_rows == c).sum())}" for c in np.unique(cyc_rows)), flush=True)

DIRS = (('HR->P', bHR, bP), ('P->HR', bP, bHR))

gate = {}
for dirn, drv, tgt in DIRS:
    gate[dirn] = estimate_gate(drv, tgt, I1, I2, dirn)
    print(f"Teil 1 fertig: {dirn}", flush=True)

res2 = {}
for dirn, drv, tgt in DIRS:
    for p, pname in ((1, 'P1'), (2, 'P2'), (3, 'P3')):
        ridx = np.where(ph_rows == p)[0]
        res2[(dirn, pname)] = estimate_phase(drv, tgt, ridx, cald_rows[ridx],
                                             cyc_rows[ridx], dirn, pname)
        print(f"Teil 2 fertig: {dirn} {pname}", flush=True)

# ------------------------------ console: part 1 ---------------------------
print(f"\nTEIL 1 - TORWAECHTER (Interaktionsmodell, voller Datensatz, "
      f"P3 = Referenz, delta = {DELTA}d):")
print(f"{'dir':6} {'n':>5} {'lam*':>8} {'R2':>5}  "
      f"{'G1(K)':>7} {'[5%':>7} {'95%]':>7}  {'G1tail':>7} {'[5%':>7} {'95%]':>7}  "
      f"{'G2(K)':>7} {'[5%':>7} {'95%]':>7}  {'G2tail':>7} {'[5%':>7} {'95%]':>7}")
for dirn, _, _ in DIRS:
    r = gate[dirn]
    stars = ['*' if lo > 0 or hi < 0 else ' '
             for (_, lo, hi) in (r['G1K'], r['G1T'], r['G2K'], r['G2T'])]
    v = []
    for (val, lo, hi), s in zip((r['G1K'], r['G1T'], r['G2K'], r['G2T']), stars):
        v.append(f"{val:+7.3f} {lo:+7.3f} {hi:+7.3f}{s}")
    print(f"{r['dir']:6} {r['n']:5d} {r['lam']:8.1e} {r['r2']:5.2f}  "
          + " ".join(v)
          + ('  <-- LAMBDA GRID EDGE!' if r['edge'] else ''))

print(f"\nLOEO (je Halving-Zyklus mit >= {MIN_ROWS} Zeilen komplett raus, "
      f"Refit bei lam*, Vorzeichen-Stabilitaet):")
for dirn, _, _ in DIRS:
    r = gate[dirn]
    parts = []
    for c, g1k, g2k in loeo_gate(r, cyc_rows):
        yr = (GENESIS + timedelta(days=int(HALVINGS[c]))).year
        parts.append(f"ohne C{c}({yr}): G1={g1k:+.2f} G2={g2k:+.2f}")
    print(f"  {dirn:6} voll: G1={r['G1K'][0]:+.2f} G2={r['G2K'][0]:+.2f} | "
          + " | ".join(parts))

# ------------------------------ console: part 2 ---------------------------
print(f"\nTEIL 2 - GETRENNTE PHASEN-KERNELS (gepoolt ueber Zyklen, "
      f"Konditionierung auf Ziel-Phase, ohne AR-Block):")
print(f"{'dir':6} {'phase':5} {'n':>5} {'nSeg':>4} {'nValZyk':>7} {'lam*':>8}  "
      f"{'H(200)':>8} {'[5%':>8} {'95%]':>8} {'H_std':>7}")
for dirn, _, _ in DIRS:
    for pname in ('P1', 'P2', 'P3'):
        r = res2[(dirn, pname)]
        print(f"{r['dir']:6} {r['phase']:5} {r['n']:5d} {r['nseg']:4d} "
              f"{r['nval']:7d} {r['lam']:8.1e}  "
              f"{r['H'][K]:+8.3f} {r['H_lo'][K]:+8.3f} {r['H_hi'][K]:+8.3f} "
              f"{r['Hstd']:+7.3f}"
              + ('  <-- LAMBDA GRID EDGE!' if r['edge'] else ''))

print("\nDifferenzen H(200) zwischen Phasen (getrennte Fits -> gemeinsame")
print("Bootstrap-Ziehungen NICHT moeglich; disjunkte Einzelbaender sind daher")
print("nur ein EXPLORATIVER Hinweis, kein gemeinsamer Differenz-Test):")
for dirn, _, _ in DIRS:
    for pa, pb in (('P1', 'P3'), ('P2', 'P3'), ('P1', 'P2')):
        ra, rb = res2[(dirn, pa)], res2[(dirn, pb)]
        dif = ra['H'][K] - rb['H'][K]
        loa, hia = ra['H_lo'][K], ra['H_hi'][K]
        lob, hib = rb['H_lo'][K], rb['H_hi'][K]
        disjoint = (hia < lob) or (hib < loa)
        verdict = ('disjunkt -> explorativer Hinweis (kein Differenz-Test)' if disjoint
                   else 'ueberlappen -> kein belastbarer Unterschied')
        print(f"  {dirn:6} H_{pa}-H_{pb} = {dif:+7.3f}   "
              f"Baender [{loa:+.3f},{hia:+.3f}] vs [{lob:+.3f},{hib:+.3f}]"
              f"  -> {verdict}")

# ------------------------------ plot: 1x2 ---------------------------------
PCOL = {'P1': '#3aa05a', 'P2': '#c0473a', 'P3': '#3a6bc0'}   # lead_lag_evol colors
PLAB = {'P1': 'P1 (100-550d)', 'P2': 'P2 (550-950d)',
        'P3': 'P3 (950d-100d, wraps halving)'}
DIRTEX = {'HR->P': r'HR$\to$P', 'P->HR': r'P$\to$HR'}

ks = np.arange(K + 1)
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 2)
axs = [fig.add_subplot(gs[0, j]) for j in range(2)]
for j, (dirn, _, _) in enumerate(DIRS):
    a = axs[j]
    a.set_facecolor('#1a1a1a')
    a.axhline(0, color='gray', linewidth=0.8, alpha=0.6)
    # grey zone k <= delta: overlapping measurement windows, no causal reading
    a.axvspan(0, DELTA, color='gray', alpha=0.12)
    for pname in ('P1', 'P2', 'P3'):
        r = res2[(dirn, pname)]
        c = PCOL[pname]
        a.fill_between(ks, r['H_lo'], r['H_hi'], color=c, alpha=0.15)
        a.plot(ks, r['H'], color=c, linewidth=2, label=PLAB[pname])
    a.set_title(rf'\textbf{{step response $H(k)$ --- {DIRTEX[dirn]}}}',
                fontsize=12, pad=4)
    a.set_xlabel('lag k (days)')
    a.legend(loc='upper left', fontsize=10, facecolor='#1A1A1A',
             edgecolor='#808080', labelcolor='#E0E0E0')
axs[0].set_ylabel(r'$H(k)$')

with plt.rc_context({'text.usetex': False}):
    plt.suptitle('Hashrate <-> Price Impulse Response - Halving-Phase Kernels',
                 color='#CCCCCC', fontsize=14, y=0.985,
                 fontweight='bold')
plt.figtext(0.5, 0.935, r'halving clock: P1 100-550d, P2 550-950d, P3 wraps '
            r'the halving --- $\delta$=11d, pooled over cycles --- '
            'pointwise 5--95\\% bands',
            ha='center', va='top', color='#999999', fontsize=10)
plt.subplots_adjust(top=0.84, bottom=0.10, left=0.06, right=0.97, wspace=0.16)
plt.savefig(os.path.join(BASE, 'impulse_phase_nHP.png'), dpi=300,
            facecolor='#0a0a0a')

# ------------------------------ reading guide -----------------------------
print("\nREADING GUIDE:")
print(" - Sterne/Band-Ausschluss von 0: punktweise nominale 90%-Baender bei festem")
print("   lambda*, explorativ - keine simultane Signifikanz ueber Lags/Phasen.")
print(" - LOEO ist ein Sensitivitaetscheck, keine formale Absicherung gegen")
print("   Episoden-Anekdotik (nur ~4 nutzbare Zyklen).")
print(" - TORWAECHTER ZUERST: nur wenn G1/G2 (Teil 1) anschlagen, sind die")
print("   getrennten Phasen-Kernels (Teil 2) als Phasen-UNTERSCHIED lesbar -")
print("   sonst zeigen sie nur die gemeinsame Basis-Antwort in anderer Aufteilung.")
print(f" - Grauzone k <= {DELTA}: Messfenster von Treiber und Ziel ueberlappen")
print("   (common shock) - kausale Lesart erst ab k > delta (k >= delta+1).")
print(" - Baender: punktweise, nominal, bedingt auf lambda* (Modellwahl nicht")
print("   mitgebootstrappt); block-wild (Vorzeichen-Flip in place).")
print(" - Teil 2 ohne AR-Block = unkonditionierte Phasen-Antwort, vergleichbar")
print("   mit H(200) der Hauptanalyse; Teil 1 misst das EXTRA relativ zu P3.")

plt.show()
