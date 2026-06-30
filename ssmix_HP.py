#!/usr/bin/env python3
"""ssmix_HP.py -- Scale-Mixture State-Space model for the BTC exponents (incremental).

ssmix = State-Space + scale-MIXture.  HP = Hashrate & Price.  Full model: ssmix_model.md.

STEP 1+2 (this version): GAUSSIAN BASELINE on BOTH series.
Local-linear-trend (integrated random walk) Kalman filter + RTS smoother on the
PRICE exponent n_P AND the HASHRATE exponent n_HR. Validation goal: each latent
trend z(t) reproduces SG-365 -- but WITH an uncertainty band (which SG-365 lacks).
This proves the state-space machinery BEFORE we make the observation model robust
(Laplace for price / Student-t for hashrate) in later steps.

State:  a_t = [z_t, v_t]   (level + velocity)
        z_t = z_{t-1} + v_{t-1}            (IRW: no separate level noise)
        v_t = v_{t-1} + w_t,  w_t ~ N(0, Q_v)
Obs:    n_t = z_t + eps_t,  eps_t ~ N(0, R)   (Gaussian for now)
Smoothing strength via an equivalent bandwidth h:  Q_v = R / h^4.

Run:  python ssmix_HP.py [--SG 365] [--bw 45] [--band latent|obs|both] [--series P|H|both]
"""
import os
import sys
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- reach the parent (zeitgeist) for the shared modules + data ---
PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT)
from data_io import load_cm, positive, days_since_genesis
from dailynHR import daily_exponent, DATA_FILE as _DF, STYLE_FILE as _SF

DATA_FILE = _DF if os.path.isabs(_DF) and os.path.exists(_DF) else os.path.join(PARENT, os.path.basename(_DF))
STYLE_FILE = _SF if os.path.isabs(_SF) and os.path.exists(_SF) else os.path.join(PARENT, os.path.basename(_SF))

# ===================== GOAL — READ THIS BEFORE EDITING =====================
# We IMPROVE the reference Kalman (Gaussian, LEVEL space) in several ways:
#   * EXPONENT space (n_P, n_H) -> stationary I(0), no spurious-regression danger
#   * MEASURED noise, NOT Gaussian: Laplace (price, b~102) / Student-t PER CYCLE
#     (hashrate, sigma/nu per halving cycle -- sec 3b of ssmix_model.md)
#   * demodulate signal vs noise -> a next-day state estimator (FILTER) + the
#     price<->hashrate coupling gamma(t) (M2, ssmix_TVP.py)
#
# THE SIGNAL  z = SG-365(n)  IS OUR SIGNAL (definition, fixed). The smoother
# ESTIMATES z USING the measured noise CHANNEL, then we VERIFY z ~ SG per cycle.
# >>> Do NOT move z off SG as "a new robust signal" -- that was a past mistake. <<<
# >>> Do NOT use a GLOBAL noise scale -- use the per-cycle measured values.   <<<
# Full, current spec + decision log: ssmix_model.md (top, "CURRENT STATE").
# ===========================================================================

# ------------------------------ parameters -------------------------------
SG_WIN    = 365      # reference Savitzky-Golay window [days]
COLOR_TXT = '#ecdcc4'
COL_RAW   = '#5a5a5a'   # raw daily exponent (faint)
COL_SG    = '#e8b84b'   # SG reference
COL_Z     = '#6db3f2'   # ssmix latent z
COL_OBS   = '#9b7fb0'   # observation (data-envelope) band
HALVINGS  = {'H1': 1425, 'H2': 2744, 'H3': 4146, 'H4': 5586}   # halving days
SERIES    = {'P': ('P', 'price',    r'n_P', r'z_P'),
             'H': ('H', 'hashrate', r'n_H', r'z_H')}
# -------------------------------------------------------------------------


def load_n(series, sg):
    """exponent of `series` ('P' or 'H') on an integer-day grid + its SG-`sg` reference."""
    d = load_cm(DATA_FILE)
    t = days_since_genesis(d['date'])
    X, tX = positive(d[series], t)
    n, dn = daily_exponent(X, tX)
    g = np.arange(int(dn.min()), int(dn.max()) + 1)
    n_grid = np.interp(g, dn, n)
    w = sg + 1 if sg % 2 == 0 else sg
    return g, n_grid, savgol_filter(n_grid, w, 2)


def kalman_smoother(y, R, Qhi, order=1):
    """Local-polynomial-trend Kalman filter + RTS smoother (integrated random walk).

    order=1: state [level, velocity]            -> cubic spline   (old local-linear).
    order=2: state [level, velocity, accel]     -> quintic spline (local-quadratic).
    Process noise sits on the HIGHEST derivative (Qhi). This only changes the LATENT
    estimate z (its flexibility at turns) -- SG-365 (the reference signal) is untouched.

    R may be a scalar or a per-point array. Returns (z, velocity, sz).
    """
    n = len(y); d = order + 1
    F = np.eye(d)                                    # Taylor / kinematic transition
    for i in range(d):
        for j in range(i + 1, d):
            F[i, j] = 1.0 / math.factorial(j - i)    # rows: 1,1,1/2 ; 0,1,1 ; 0,0,1
    Q = np.zeros((d, d)); Q[-1, -1] = Qhi            # noise on highest derivative
    H = np.zeros((1, d)); H[0, 0] = 1.0
    Id = np.eye(d)
    Rt = R if np.ndim(R) else np.full(n, float(R))

    a_pred = np.zeros((n, d)); P_pred = np.zeros((n, d, d))
    a_filt = np.zeros((n, d)); P_filt = np.zeros((n, d, d))

    a = np.zeros(d); a[0] = y[0]
    P = np.eye(d) * 1e6                              # diffuse init
    for t in range(n):
        ap = F @ a; Pp = F @ P @ F.T + Q            # predict
        a_pred[t] = ap; P_pred[t] = Pp
        S = (H @ Pp @ H.T)[0, 0] + Rt[t]            # update
        K = (Pp @ H.T).ravel() / S
        a = ap + K * (y[t] - (H @ ap)[0])
        P = (Id - np.outer(K, H.ravel())) @ Pp
        a_filt[t] = a; P_filt[t] = P

    a_s = a_filt.copy(); P_s = P_filt.copy()        # RTS backward pass
    for t in range(n - 2, -1, -1):
        C = P_filt[t] @ F.T @ np.linalg.solve(P_pred[t + 1], Id)
        a_s[t] = a_filt[t] + C @ (a_s[t + 1] - a_pred[t + 1])
        P_s[t] = P_filt[t] + C @ (P_s[t + 1] - P_pred[t + 1]) @ C.T

    return a_s[:, 0], a_s[:, 1], np.sqrt(np.maximum(P_s[:, 0, 0], 0.0))


# MEASURED observation laws, PER CYCLE (see ssmix_model.md sec 3b) -- TWO tables:
NOISE = {'P': dict(dist='laplace'),     # price: Laplace per cycle (PRICE_CYC)
         'H': dict(dist='student')}     # hashrate: Student-t per cycle (HASH_CYC)

# price n_P    -- Laplace per cycle: (start_day, b);           pre-'13 -> '13
PRICE_CYC = [(1425, 57.0), (2744, 96.0), (4146, 107.0), (5586, 103.0)]
# hashrate n_H -- Student-t per cycle: (start_day, sigma, nu); pre-'13 -> '13
HASH_CYC  = [(1425, 219.0, 17.5), (2744, 377.0, 9.2), (4146, 566.0, 21.8), (5586, 697.0, 158.8)]


def price_cycle_b(g):
    """per-day Laplace scale b from the MEASURED per-cycle price fits (sec 3b)."""
    g = np.asarray(g)
    b = np.full(len(g), PRICE_CYC[0][1])
    for start, bb in PRICE_CYC:
        b[g >= start] = bb
    return b


def hash_cycle_noise(g):
    """per-day Student-t scale s and dof nu from the MEASURED per-cycle fits (sec 3b)."""
    g = np.asarray(g)
    s = np.full(len(g), HASH_CYC[0][1]); nu = np.full(len(g), HASH_CYC[0][2])
    for start, sig, nuc in HASH_CYC:
        m = g >= start
        s[m] = sig; nu[m] = nuc
    return s, nu


def robust_smoother(series, g, n, sg_n, bw, order, n_iter=12):
    """Robust scale-mixture IRLS Kalman smoother with the MEASURED noise channel:
    Laplace PER-CYCLE for price, Student-t PER-CYCLE for hashrate. The noise channel
    is FIXED to the measured values -- not re-fitted; we estimate z with it and (in
    main) check z against SG per cycle. `order` = latent-trend order (1 or 2)."""
    cfg = NOISE[series]
    Qhi_exp = 2 * (order + 1)                            # Q on highest derivative
    if cfg['dist'] == 'laplace':
        b = price_cycle_b(g)                             # per-cycle Laplace scale
        R_scale = float(np.median(2.0 * b * b))
        Qhi = R_scale / bw ** Qhi_exp
        z = sg_n.copy()
        for _ in range(n_iter):
            r = n - z
            Rt = b * np.abs(r) + 1e-3 * R_scale          # E[1/tau^2] ~ 1/(b|r|)
            z, v, sz = kalman_smoother(n, Rt, Qhi, order)
        w = R_scale / Rt
    else:                                                # Student-t, per-cycle s, nu
        s, nu = hash_cycle_noise(g)
        R_scale = float(np.median(s ** 2 * nu / np.maximum(nu - 2.0, 1.0)))
        Qhi = R_scale / bw ** Qhi_exp
        z = sg_n.copy()
        for _ in range(n_iter):
            r = n - z
            Rt = s ** 2 * (nu + (r / s) ** 2) / (nu + 1.0)
            z, v, sz = kalman_smoother(n, Rt, Qhi, order)
        w = s ** 2 / Rt
    return dict(z=z, v=v, sz=sz, w=w / np.median(w), R=R_scale, dist=cfg['dist'])


def fit_series(series, sg, bw, obs, order):
    """Run the smoother (Gaussian baseline or robust) on one series."""
    g, n, sg_n = load_n(series, sg)
    if obs == 'robust':
        r = robust_smoother(series, g, n, sg_n, bw, order)
    else:
        R = float(np.var(n - sg_n))
        z, v, sz = kalman_smoother(n, R, R / bw ** (2 * (order + 1)), order)
        r = dict(z=z, v=v, sz=sz, w=np.ones(len(n)), R=R, dist='gauss')
    h = sg // 2                                       # drop IRW edge-extrapolation zone
    sl = slice(h, len(g) - h)
    return dict(g=g[sl], n=n[sl], sg_n=sg_n[sl], R=r['R'], dist=r['dist'],
                z=r['z'][sl], v=r['v'][sl], sz=r['sz'][sl], w=r['w'][sl],
                rmse=float(np.sqrt(np.mean((r['z'][sl] - sg_n[sl]) ** 2))))


def plot_panel(ax, res, key, band, sg, bw, panel_tag):
    """Draw one series panel."""
    col, name, sym, zsym = SERIES[key]
    g, n, sg_n, z, sz, R = res['g'], res['n'], res['sg_n'], res['z'], res['sz'], res['R']
    sz_obs = np.sqrt(sz ** 2 + R)

    ax.scatter(g, n, s=3, color=COL_RAW, alpha=0.35, label=f'raw ${sym}$ (daily)')
    if band in ('obs', 'both'):
        ax.fill_between(g, z - 2 * sz_obs, z + 2 * sz_obs, color=COL_OBS, alpha=0.18,
                        label=r'obs band $\pm 2\sqrt{\sigma_z^2+R}$')
    ax.plot(g, sg_n, color=COL_SG, lw=1.8, label=f'SG-{sg}')
    if band in ('latent', 'both'):
        ax.fill_between(g, z - 2 * sz, z + 2 * sz, color=COL_Z, alpha=0.32,
                        label=fr'latent band ${zsym}\pm 2\sigma_z$')
    ax.plot(g, z, color=COL_Z, lw=1.8, label=f'ssmix ${zsym}$')

    if band in ('obs', 'both'):
        lo, hi = np.percentile(n, [1, 99])
        ax.set_ylim(min(lo, float((z - 2 * sz_obs).min())),
                    max(hi, float((z + 2 * sz_obs).max())))
    else:
        lo = min(float((z - 2 * sz).min()), float(sg_n.min()))
        hi = max(float((z + 2 * sz).max()), float(sg_n.max()))
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

    y0, y1 = ax.get_ylim()
    for lab, dday in HALVINGS.items():
        ax.axvline(dday, color='#6f5d46', lw=0.8, ls='--', zorder=1)
        ax.text(dday, y1 - 0.05 * (y1 - y0), f" {lab}", color='#9b8a6f',
                fontsize=9, va='top', ha='left')
    ax.set_title(f'{panel_tag} {name} exponent  ${sym}$', color=COLOR_TXT, fontsize=12)
    ax.text(0.013, 0.95, f'SG-{sg}   bw={bw:.0f}d   obs: {res["dist"]}\nRMSE(z,SG) = {res["rmse"]:.2f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=8.5, color='#cfc2a8',
            bbox=dict(boxstyle='round', facecolor='#1A1A1A', edgecolor='#6f6f6f', alpha=0.85))
    ax.set_ylabel(f'${sym}$')
    ax.grid(True, alpha=0.3, ls='--')
    ax.legend(loc='upper right', fontsize=9, facecolor='#1A1A1A',
              edgecolor='#808080', labelcolor='#E0E0E0', ncol=2)


def main():
    ap = argparse.ArgumentParser(description='ssmix step 1+2: Gaussian IRW Kalman smoother on n_P and n_HR')
    ap.add_argument('--SG', type=int, default=SG_WIN,
                    help='filter width [days]; sets SG reference and Kalman smoothing (bw=SG/8)')
    ap.add_argument('--bw', type=float, default=None, help='override Kalman bandwidth [days] (default SG/8)')
    ap.add_argument('--band', choices=['latent', 'obs', 'both'], default='latent',
                    help='latent = signal uncertainty; obs = data-envelope band; both')
    ap.add_argument('--series', choices=['P', 'H', 'both'], default='both',
                    help='which exponent(s) to show (default both: price top, hashrate bottom)')
    ap.add_argument('--obs', choices=['gauss', 'robust'], default='gauss',
                    help='observation law: gauss baseline (matches SG), or robust (Laplace price / Student-t hashrate)')
    ap.add_argument('--poly', type=int, choices=[1, 2], default=2,
                    help='latent-trend order: 1 = local-linear (level+velocity), '
                         '2 = local-quadratic (+acceleration). Only changes z, NOT SG.')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    bw = args.bw if args.bw is not None else args.SG / 8.0
    keys = ['P', 'H'] if args.series == 'both' else [args.series]
    res = {k: fit_series(SERIES[k][0], args.SG, bw, args.obs, args.poly) for k in keys}

    CYC_B = {'13': (1425, 2744), '17': (2744, 4146), '21': (4146, 5586), '25': (5586, 99999)}
    model = ('GAUSS baseline (constant noise, z = SG)' if args.obs == 'gauss'
             else 'MEASURED per-cycle noise (Laplace price / Student-t hashrate)')
    print("============= SIGNAL CHECK (smoother, looking back) =============")
    print("We estimate the smooth signal z and compare it to SG (= our signal).")
    print(f"Noise model used: {model}")
    print(f"latent-trend order: poly-{args.poly} "
          f"({'local-linear' if args.poly == 1 else 'local-quadratic'})  [SG untouched]")
    print("'z within +-X of SG' = how close the estimate is to SG (small = good).")
    print("'band' = the +-2 sigma uncertainty around z.")
    print("================================================================\n")
    for k in keys:
        r = res[k]
        print(f"{SERIES[k][1].upper()} exponent:")
        for c, (a, b) in CYC_B.items():            # per-cycle: how close is z to SG?
            m = (r['g'] >= a) & (r['g'] < b)
            if m.sum() > 30:
                rc = float(np.sqrt(np.mean((r['z'][m] - r['sg_n'][m]) ** 2)))
                ok = 'z matches SG well' if rc < 10 else 'z drifts from SG'
                print(f"   cycle '{c}:  z within +-{rc:4.1f} of SG    band +-{2*np.mean(r['sz'][m]):5.1f}"
                      f"    ->  {ok}")
        print()
    if args.no_plot:
        return

    plt.style.use(STYLE_FILE)
    tags = ['(a)', '(b)']
    if len(keys) == 2:
        fig, axes = plt.subplots(2, 1, figsize=(13, 7)); axes = list(axes)
    else:
        fig, ax0 = plt.subplots(figsize=(13, 5)); axes = [ax0]
    for ax, k, tag in zip(axes, keys, tags):
        plot_panel(ax, res[k], k, args.band, args.SG, bw, tag)
    axes[-1].set_xlabel('days since genesis')

    plt.suptitle('ssmix — Gaussian state-space smoother of the BTC exponents',
                 color=COLOR_TXT, fontsize=14, y=0.985, fontweight='bold')
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, hspace=0.25)
    tag = 'PH' if len(keys) == 2 else keys[0]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f'ssmix_step1_{tag}_SG{args.SG}_{args.band}.png')
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"saved {out}")
    plt.show()


if __name__ == '__main__':
    main()
