#!/usr/bin/env python3
"""SG-365d drift of the hashrate exponent n_HR(t) and the price exponent n_P(t),
overlaid in one panel. Only the 365-day Savitzky-Golay curve is shown (the
shorter filters are omitted) -- this is the model-consistent view for comparing
the two slow drifts and the precursor to the drift-correlation test.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from scipy.signal import savgol_filter

from data_io import load_cm, positive, days_since_genesis
from dailynHR import daily_exponent, DATA_FILE

# ------------------------------ parameters -------------------------------
HERE        = os.path.dirname(os.path.abspath(__file__))
STYLE_FILE  = os.path.join(HERE, 'public.mplstyle')
OUT_PNG     = os.path.join(HERE, 'dailyn_HP_evol.png')

SG_WIN      = 365               # only the 365-day Savitzky-Golay window
POLYORDER   = 2

HALVINGS    = {'H1': 1425, 'H2': 2744, 'H3': 4146, 'H4': 5586}

FIGSIZE     = (12, 7)
COLOR_HALV  = '#b3a081'
COLOR_ZERO  = '#6f5d46'
COLOR_TXT   = '#ecdcc4'
COLOR_SUB   = '#b3a081'
COLOR_H     = '#6db3f2'         # hashrate exponent (blue)
COLOR_P     = '#e8b84b'         # price exponent (gold)
# -------------------------------------------------------------------------


def to_odd(w):
    """Nearest odd integer >= 3 (Savitzky-Golay requires an odd window)."""
    w = int(round(w))
    if w % 2 == 0:
        w += 1
    return max(w, 3)


def sg365(series_key, sg_win):
    """SG drift of the daily exponent of series `series_key` (H or P),
    edges (+/- win/2) cropped to remove the SG extrapolation."""
    d = load_cm(DATA_FILE)
    t_age = days_since_genesis(d['date'])
    X, t = positive(d[series_key], t_age)
    n, day = daily_exponent(X, t)
    o = np.argsort(day)
    n, day = n[o], day[o]
    win = to_odd(sg_win)
    c = savgol_filter(n, win, POLYORDER)
    h = win // 2
    return day[h:-h], c[h:-h]


def draw_halvings(ax):
    """Vertical dashed guides + labels at the four halving days."""
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for name, x in HALVINGS.items():
        ax.axvline(x, color=COLOR_HALV, lw=1.0, ls='--', alpha=0.6, zorder=2)
        ax.text(x, 0.025, name, transform=tr, ha='center', va='bottom',
                color=COLOR_HALV, fontsize=10, fontweight='bold')


def main():
    import argparse
    ap = argparse.ArgumentParser(description='SG drift overlay n_HR vs n_P')
    ap.add_argument('--no-plot', action='store_true', help='compute only')
    ap.add_argument('--SG-H', type=int, default=365,
                    help='Savitzky-Golay width for HASHRATE n_HR [days]')
    ap.add_argument('--SG-P', type=int, default=365,
                    help='Savitzky-Golay width for PRICE n_P [days]')
    args = ap.parse_args()

    wH = to_odd(args.SG_H)
    wP = to_odd(args.SG_P)
    dayH, cH = sg365('H', args.SG_H)
    dayP, cP = sg365('P', args.SG_P)
    print(f"n_HR SG-{wH}d: N={len(dayH)}  range [{cH.min():.1f}, {cH.max():.1f}]")
    print(f"n_P  SG-{wP}d: N={len(dayP)}  range [{cP.min():.1f}, {cP.max():.1f}]")
    if args.no_plot:
        return

    plt.style.use(STYLE_FILE)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhline(0.0, color=COLOR_ZERO, lw=0.8, zorder=1)
    draw_halvings(ax)
    ax.plot(dayH, cH, color=COLOR_H, lw=2.4, zorder=4,
            label=rf'Hashrate  $n_{{HR}}$  (SG-{wH}d)')
    ax.plot(dayP, cP, color=COLOR_P, lw=2.4, zorder=4,
            label=rf'Price  $n_P$  (SG-{wP}d)')

    ax.set_xlim(min(dayH.min(), dayP.min()), max(dayH.max(), dayP.max()))
    ax.set_xlabel('Days since genesis block')
    ax.set_ylabel(r'$n$  (power-law exponent)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', labelcolor=COLOR_TXT)

    plt.suptitle('Bitcoin: Hashrate vs Price Power-Law Exponent Drift',
                 color=COLOR_TXT, fontsize=14, y=0.975, fontweight='bold')
    plt.figtext(0.5, 0.935, f'SG-{wH}d (H) / SG-{wP}d (P) filtered drift of the daily exponents',
                ha='center', va='top', color=COLOR_SUB, fontsize=10)
    plt.subplots_adjust(top=0.90, bottom=0.09, left=0.08, right=0.97)
    fig.savefig(OUT_PNG, dpi=300, facecolor=fig.get_facecolor())
    print(f"saved {OUT_PNG}")
    plt.show()


if __name__ == '__main__':
    main()
