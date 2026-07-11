#!/usr/bin/env python3
"""ssmix_estimator.py -- the next-day state estimator (FILTER reading of ssmix).

# ===================== GOAL — READ THIS BEFORE EDITING =====================
# This is the FILTER (causal, forward-only) reading of the ssmix model: a
# one-step-ahead estimator for TOMORROW's exponent. At each day it predicts
#   z_{t|t-1} = z_{t-1|t-1} + v_{t-1|t-1}        (signal carried forward by its slope)
# and a PREDICTIVE BAND from the MEASURED per-cycle noise channel (Laplace price /
# Student-t hashrate, sec 3b of ssmix_model.md). The fraction of actual n_t that
# falls inside the band (coverage, target ~95%) VERIFIES the noise model.
#
# Signal stays SG-anchored; the noise channel is FIXED to the measured per-cycle
# values (NOT re-fitted). Smoother = retrospective (ssmix_HP.py); FILTER = this =
# the next-day estimator. Full spec: ssmix_model.md.
# ===========================================================================

Run:  python ssmix_estimator.py [--SG 365] [--bw 46] [--series P|H|both]
"""
import os
import sys
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_io import load_cm, positive, days_since_genesis
from ssmix_HP import (load_n, NOISE, price_cycle_b, hash_cycle_noise,
                      STYLE_FILE, HALVINGS, COLOR_TXT, SG_WIN, SERIES, DATA_FILE)


def level_on_grid(col, g):
    """actual price/hashrate level on the integer-day grid g (log-interpolated)."""
    d = load_cm(DATA_FILE)
    t = days_since_genesis(d['date'])
    X, tX = positive(d[col], t)
    return np.exp(np.interp(g, tX, np.log(X)))

COL_RAW = '#5a5a5a'
COL_Z   = '#6db3f2'
COL_BAND = '#9b7fb0'
CYC_B = {'13': (1425, 2744), '17': (2744, 4146), '21': (4146, 5586), '25': (5586, 99999)}


def robust_filter(series, g, n, bw, order=2):
    """Causal robust Kalman FILTER (forward only) with the measured per-cycle noise
    channel. `order` = latent-trend order (1 local-linear, 2 local-quadratic, like
    ssmix_HP). Returns one-step-ahead prediction zp = z_{t|t-1} and the per-cycle 95%
    predictive half-width q."""
    N = len(n); d = order + 1
    if NOISE[series]['dist'] == 'laplace':                 # price
        b = price_cycle_b(g)
        R_scale = float(np.median(2.0 * b * b))
        q = 2.996 * b                                      # 95% half-width of Laplace(0,b)
        Rt_fun = lambda r, t: b[t] * abs(r) + 1e-3 * R_scale
    else:                                                  # hashrate, Student-t
        s, nu = hash_cycle_noise(g)
        R_scale = float(np.median(s ** 2 * nu / np.maximum(nu - 2.0, 1.0)))
        q = stats.t.ppf(0.975, nu) * s                     # 95% half-width of t_nu(0,s)
        Rt_fun = lambda r, t: s[t] ** 2 * (nu[t] + (r / s[t]) ** 2) / (nu[t] + 1.0)
    Qhi = R_scale / bw ** (2 * (order + 1))

    F = np.eye(d)                                          # Taylor / kinematic transition
    for i in range(d):
        for j in range(i + 1, d):
            F[i, j] = 1.0 / math.factorial(j - i)
    Q = np.zeros((d, d)); Q[-1, -1] = Qhi
    H = np.zeros((1, d)); H[0, 0] = 1.0; Id = np.eye(d)
    a = np.zeros(d); a[0] = n[0]; P = np.eye(d) * 1e6
    zp = np.zeros(N); sp = np.zeros(N)
    for t in range(N):
        ap = F @ a; Pp = F @ P @ F.T + Q                   # predict -> z_{t|t-1}
        zp[t] = ap[0]; sp[t] = np.sqrt(max(Pp[0, 0], 0.0))
        r = n[t] - ap[0]                                   # causal one-step residual
        Rt = Rt_fun(r, t)
        S = Pp[0, 0] + Rt; K = (Pp @ H.T).ravel() / S      # update with the new obs
        a = ap + K * r; P = (Id - np.outer(K, H.ravel())) @ Pp
    return zp, q, sp


def main():
    ap = argparse.ArgumentParser(description='ssmix next-day state estimator (causal filter)')
    ap.add_argument('--SG', type=int, default=SG_WIN)
    ap.add_argument('--bw', type=float, default=None)
    ap.add_argument('--series', choices=['P', 'H', 'both'], default='both')
    ap.add_argument('--poly', type=int, choices=[1, 2], default=2,
                    help='latent-trend order: 1 local-linear, 2 local-quadratic (like ssmix_HP)')
    ap.add_argument('--space', choices=['exp', 'level'], default='level',
                    help='exp = exponent space (wide band); level = price/hashrate space (narrow forecast band)')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()
    bw = args.bw if args.bw is not None else args.SG / 8.0
    keys = ['P', 'H'] if args.series == 'both' else [args.series]

    res = {}
    print("=================== NEXT-DAY ESTIMATOR ===================")
    print("Each day we predict tomorrow's exponent and draw a 95% band")
    print("from the MEASURED per-cycle noise. If the noise model is")
    print("right, about 95% of the real next-day values land in the band.")
    print("=========================================================\n")
    for k in keys:
        col = SERIES[k][0]
        g, n, _ = load_n(col, args.SG)
        zp, q, sp = robust_filter(col, g, n, bw, args.poly)
        h = args.SG // 2; sl = slice(h, len(g) - h)        # drop warm-up / edge zone
        g, n, zp, q = g[sl], n[sl], zp[sl], q[sl]
        res[k] = dict(g=g, n=n, zp=zp, q=q,
                      rmse=float(np.sqrt(np.mean((n - zp) ** 2))))
        noise = 'Laplace' if col == 'P' else 'Student-t'
        print(f"{SERIES[k][1].upper()} exponent  ({noise} noise per cycle):")
        for c, (a, b) in CYC_B.items():
            m = (g >= a) & (g < b)
            if m.sum() > 30:
                cov = float(np.mean(np.abs(n[m] - zp[m]) <= q[m])) * 100
                verdict = ('noise model FITS' if 90 <= cov <= 98
                           else 'band too WIDE' if cov > 98 else 'band too NARROW')
                print(f"   cycle '{c}:  the 95% band catches {cov:4.1f}% of real days  ->  {verdict}")
        print()
    if args.no_plot:
        return

    plt.style.use(STYLE_FILE)
    fig, axes = plt.subplots(len(keys), 1, figsize=(13, 7 if len(keys) == 2 else 4.5))
    axes = np.atleast_1d(axes)
    for ax, k, tag in zip(axes, keys, ['(a)', '(b)']):
        r = res[k]; col, name, sym, _ = SERIES[k]
        if args.space == 'level':                          # price/hashrate space (narrow)
            X = level_on_grid(col, r['g'])
            ratio = (r['g'] + 1.0) / r['g']                # back-transform: X*(t+1/t)^(z+-q)
            ax.fill_between(r['g'], X * ratio ** (r['zp'] - r['q']), X * ratio ** (r['zp'] + r['q']),
                            color=COL_BAND, alpha=0.35, label='95% next-day band')
            ax.plot(r['g'], X, color='#888888', lw=0.9, alpha=0.8, label=f'real {name}')
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_ylabel(name)
            ax.set_title(f'{tag} {name}  (next-day forecast band)', color=COLOR_TXT, fontsize=12)
        else:                                              # exponent space (wide)
            ax.scatter(r['g'], r['n'], s=3, color=COL_RAW, alpha=0.35, label=f'actual ${sym}$')
            ax.fill_between(r['g'], r['zp'] - r['q'], r['zp'] + r['q'], color=COL_BAND,
                            alpha=0.20, label='95% predictive band')
            ax.plot(r['g'], r['zp'], color=COL_Z, lw=1.6, label='next-day estimate')
            lo, hi = np.percentile(r['n'], [1, 99]); ax.set_ylim(lo, hi)
            ax.set_ylabel(f'${sym}$')
            ax.set_title(f'{tag} {name} exponent  ${sym}$', color=COLOR_TXT, fontsize=12)
        y0, y1 = ax.get_ylim()
        for lab, dday in HALVINGS.items():
            ax.axvline(dday, color='#6f5d46', lw=0.8, ls='--', zorder=1)
            ax.text(dday, y1, f" {lab}", color='#9b8a6f', fontsize=9, va='top', ha='left')
        ax.grid(True, which='both', alpha=0.3, ls='--')
        ax.legend(loc='upper left', fontsize=9, facecolor='#1A1A1A',
                  edgecolor='#808080', labelcolor='#E0E0E0', ncol=2)
    axes[-1].set_xlabel('days since genesis')
    title = ('ssmix — next-day forecast band in price/hashrate space' if args.space == 'level'
             else 'ssmix — next-day state estimator in exponent space')
    plt.suptitle(title, color=COLOR_TXT, fontsize=14, y=0.985, fontweight='bold')
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, hspace=0.25)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'ssmix_estimator_{args.space}.png')
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"saved {out}")
    plt.show()


if __name__ == '__main__':
    main()
