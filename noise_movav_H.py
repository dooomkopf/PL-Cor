#!/usr/bin/env python3
"""noise_movav_H.py -- MC noise-strip around a centered moving average of the HASHRATE.

Full analogy to MC/noise_movav.py (price), but for the Bitcoin HASHRATE H, here in
zeitgeist with the local modules. Goal: SEE how well our per-cycle daily-exponent
noise model reproduces the real hashrate scatter.

Anchor = centered moving average (window n) of the REAL hashrate H(t). Around it the
rolling Q(100-Q)..Q(Q) noise band is drawn, DRIFT-FREE (mu = 0 in the daily step:
only the WIDTH of the noise is used, not its location), with a PER-CYCLE noise scale
fitted from the data itself.

Noise model (math):
  daily local exponent  n_HR = ln(H_t/H_{t-1}) / ln(t/(t-1))        (H ~ t^n)
  one MC step:          ln H_t = ln H_{t-1} + n * ln(t/(t-1)),  n ~ Student-t(nu_c)*s_c
  per cycle c: (nu_c, s_c) = MLE Student-t fit of the daily n_HR in cycle c, mu=0.
A good noise model => the Q1..Q99 band envelopes the real hashrate scatter.

Run:  ./noise_movav_H.py [--window 30] [--N 500] [--Q 99] [--laplace] [--log]
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
from scipy import stats

from data_io import load_cm, positive, days_since_genesis
from dailynHR import daily_exponent, DATA_FILE, STYLE_FILE

# ------------------------------ parameters -------------------------------
WINDOW      = 30          # centered moving-average window for the anchor [days]
N_PATHS     = 500         # MC paths per anchor day
HORIZON     = 1           # MC horizon per anchor [days] (1 = daily noise cone)
Q_MAX       = 99.0        # band = [100-Q .. Q] percentile
DIST        = 'student'   # 'student' (default) or 'laplace'

# halving block-reward days since genesis -> cycle starts
HALVING_T   = {'13': 1425, '17': 2744, '21': 4146, '25': 5586}
CYCLE_ORDER = ['13', '17', '21', '25']
CYCLE_COLORS = {'13': '#0000FF', '17': '#90EE90', '21': '#FF69B4', '25': 'orange'}
H_COLOR     = '#6db3f2'   # hashrate line (blue)
# -------------------------------------------------------------------------


def cycle_for_day(t):
    """Cycle label for day t (days since genesis); days before '13 fall back to '13."""
    label = CYCLE_ORDER[0]
    for c in CYCLE_ORDER:
        if t >= HALVING_T[c]:
            label = c
        else:
            break
    return label


def load_hashrate():
    """(t, H): network age [days since genesis] and hashrate, H>0, sorted by t."""
    d = load_cm(DATA_FILE)
    t_age = days_since_genesis(d['date'])
    H, t = positive(d['H'], t_age)
    o = np.argsort(t)
    return t[o], H[o]


def fit_cycle_noise(n, day, dist):
    """Per-cycle noise scale of the daily exponent, mu IGNORED (drift-free).
    Student-t: (nu_c, s_c) by MLE.  Laplace: scale b_c = mean|x - median|."""
    lab = np.array([cycle_for_day(int(d)) for d in day])
    scale, nu = {}, {}
    for c in CYCLE_ORDER:
        x = n[lab == c]
        if len(x) < 30:
            scale[c], nu[c] = np.nan, np.nan
            continue
        if dist == 'student':
            nu_c, _loc, s_c = stats.t.fit(x)       # (df, loc, scale); loc dropped (mu=0)
            scale[c], nu[c] = s_c, nu_c
        else:
            scale[c], nu[c] = np.mean(np.abs(x - np.median(x))), np.nan
    return scale, nu


def rolling_band(days, anchor_fn, scale, nu, n_paths, horizon, q, dist, rng):
    """Drift-free MC band around the moving-average anchor (mu=0). H ~ t^n => the
    daily step is dlnH = n * dln t with n drawn from the per-cycle noise law."""
    qlo_p, qhi_p = round(100.0 - q, 6), q
    t_out, lo, hi, cyc = [], [], [], []
    for t_k in days:
        if not np.isfinite(scale[cycle_for_day(int(t_k))]):
            continue
        lnH = np.full(n_paths, np.log(anchor_fn(t_k)))
        for s in range(1, horizon + 1):
            t_cur = t_k + s
            c = cycle_for_day(int(t_cur))
            if dist == 'student':
                step = rng.standard_t(nu[c], size=n_paths) * scale[c]
            else:
                step = rng.laplace(0.0, scale[c], size=n_paths)    # mu = 0
            lnH = lnH + step * np.log(t_cur / (t_cur - 1))
        H_end = np.exp(lnH)
        t_out.append(t_k + horizon); cyc.append(cycle_for_day(int(t_k + horizon)))
        lo.append(np.percentile(H_end, qlo_p)); hi.append(np.percentile(H_end, qhi_p))
    return np.array(t_out), np.array(lo), np.array(hi), np.array(cyc)


def main():
    ap = argparse.ArgumentParser(description='MC noise-strip around a moving average of the HASHRATE')
    ap.add_argument('--window', type=int, default=WINDOW, help='centered MA window [days]')
    ap.add_argument('--N', type=int, default=N_PATHS, help='MC paths per anchor day')
    ap.add_argument('--horizon', type=int, default=HORIZON, help='MC horizon [days]')
    ap.add_argument('--Q', type=float, default=Q_MAX, help='max band quantile %% (50<Q<=100)')
    ap.add_argument('--start', type=int, default=None, help='start day; default = full range')
    ap.add_argument('--laplace', action='store_true', help='Laplace band instead of Student-t')
    ap.add_argument('--log', action='store_true', help='log x-axis')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()
    dist = 'laplace' if args.laplace else DIST

    t, H = load_hashrate()
    n, day = daily_exponent(H, t)
    scale, nu = fit_cycle_noise(n, day, dist)
    print(f"hashrate noise model ({dist}), per-cycle scale (mu=0, width only):")
    for c in CYCLE_ORDER:
        extra = f"  nu={nu[c]:.1f}" if dist == 'student' and np.isfinite(nu[c]) else ""
        print(f"  '{c}: scale={scale[c]:.1f}{extra}")

    anchor = pd.Series(H).rolling(window=args.window, center=True, min_periods=1).mean().values
    anchor_fn = lambda x: np.interp(x, t, anchor)

    t0 = int(args.start) if args.start is not None else int(t[0])
    days = np.arange(t0, int(t[-1]) - args.horizon + 1)
    t_out, blo, bhi, cyc = rolling_band(days, anchor_fn, scale, nu, args.N,
                                        args.horizon, args.Q, dist,
                                        np.random.default_rng(0))
    print(f"window={args.window}  N={args.N}  horizon={args.horizon}  days={len(days)}")
    if args.no_plot:
        return

    plt.style.use(STYLE_FILE)
    mpl.rcParams['font.sans-serif'] = ['Comfortaa', 'DejaVu Sans', 'Arial']
    fig, ax = plt.subplots(figsize=(13, 7))
    vis = t >= t0
    ax.plot(t[vis], H[vis], color=H_COLOR, lw=1.2, alpha=0.7, label='Hashrate (real)')
    for c in CYCLE_ORDER:
        m = cyc == c
        if m.any():
            ax.fill_between(t_out[m], blo[m], bhi[m], color=CYCLE_COLORS[c],
                            alpha=0.30, label=f"'{c}-cycle Noise Q{args.Q:g}")
    ax.set_yscale('log')
    ax.set_xscale('log')          # x-axis ALWAYS logarithmic (log-log)
    last = t[-1]
    ax.set_xlim(200, last + 0.15 * (last - 200))   # start at day 200
    ax.set_ylim(1e-8, 1e10)
    # finer day labels on the log x-axis (200, 300, 500, 1000, 2000, ...)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5), numticks=20))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(v)}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Days since Genesis')
    ax.set_ylabel('Hashrate')
    ax.grid(True, which='both', alpha=0.3, ls='--')
    ax.legend(loc='upper left', fontsize=10, facecolor='#1A1A1A',
              edgecolor='#808080', labelcolor='#E0E0E0')
    ax.set_title(f'Hashrate Noise Band via MC of real daily Exponents '
                 f'[{"Student-t" if dist == "student" else "Laplace"}]',
                 color='#CCCCCC', fontsize=13, fontfamily='Comfortaa',
                 fontweight='bold', usetex=False, pad=14)
    _w, _h = fig.get_size_inches()
    fig.subplots_adjust(top=1 - 0.5 / _h, bottom=0.6 / _h, left=1.1 / _w, right=1 - 0.35 / _w)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'noise_movav_H.png')
    fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    print(f"saved {out}")
    plt.show()


if __name__ == '__main__':
    main()
