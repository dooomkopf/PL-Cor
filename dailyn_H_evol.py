#!/usr/bin/env python3
"""Time-resolved drift of the daily Bitcoin hashrate exponent n_HR(t).

n_HR = log(H2/H1) / log(t2/t1)  (see dailynHR.daily_exponent). Savitzky-Golay
(SG) curves of growing window length are overlaid on the raw daily points
(coloured by halving cycle) to expose the slow drift under the heavy noise.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

from data_io import load_cm, positive, days_since_genesis
from dailynHR import daily_exponent, DATA_FILE, CYCLES, COLORS

# ------------------------------ parameters -------------------------------
HERE        = os.path.dirname(os.path.abspath(__file__))
STYLE_FILE  = os.path.join(HERE, 'public.mplstyle')
OUT_PNG     = os.path.join(HERE, 'dailyn_H_evol.png')

SG_WINDOWS  = [91, 181, 365]    # Savitzky-Golay window lengths [days] (rounded to odd)
POLYORDER   = 2                 # SG local polynomial order
YLIM        = (-75, 75)         # fixed symmetric y-range for the drift plot

# halving block-reward dates, in days since the genesis block
HALVINGS    = {'H1': 1425, 'H2': 2744, 'H3': 4146, 'H4': 5586}

FIGSIZE     = (12, 7)
COLOR_HALV  = '#b3a081'
COLOR_ZERO  = '#6f5d46'
COLOR_TXT   = '#ecdcc4'
COLOR_SUB   = '#b3a081'
COLOR_PRE   = '#777777'         # raw points before the first halving
SG_STYLE    = {91:  ('#6db3f2', 1.0, 0.75),
               181: ('#4ec9b0', 1.7, 0.95),
               365: ('#e8b84b', 2.6, 1.00)}
# -------------------------------------------------------------------------


def to_odd(w):
    """Nearest odd integer >= 3 (Savitzky-Golay requires an odd window)."""
    w = int(round(w))
    if w % 2 == 0:
        w += 1
    return max(w, 3)


def load_nHR():
    """Daily hashrate exponent n_HR and its left-day, sorted by day."""
    d = load_cm(DATA_FILE)
    t_age = days_since_genesis(d['date'])
    H, t = positive(d['H'], t_age)
    n, day = daily_exponent(H, t)
    o = np.argsort(day)
    return n[o], day[o]


def sg_curves(n, windows, polyorder):
    """Savitzky-Golay smooth of the series for each window length."""
    return {w: savgol_filter(n, to_odd(w), polyorder) for w in windows}


def draw_halvings(ax):
    """Vertical dashed guides + labels at the four halving days."""
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for name, x in HALVINGS.items():
        ax.axvline(x, color=COLOR_HALV, lw=1.0, ls='--', alpha=0.6, zorder=2)
        ax.text(x, 0.025, name, transform=tr, ha='center', va='bottom',
                color=COLOR_HALV, fontsize=10, fontweight='bold')


def make_plot(day, n_raw, curves):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhline(0.0, color=COLOR_ZERO, lw=0.8, zorder=1)

    # raw n_HR: one small dot per day, coloured by halving cycle (runs off-range)
    pre = day < min(s for s, _ in CYCLES.values())
    if pre.any():
        ax.scatter(day[pre], n_raw[pre], s=3, alpha=0.25, color=COLOR_PRE,
                   edgecolors='none', zorder=1)
    for (_name, (s, e)), col in zip(CYCLES.items(), COLORS):
        m = (day >= s) & (day < e)
        ax.scatter(day[m], n_raw[m], s=3, alpha=0.35, color=col,
                   edgecolors='none', zorder=1)

    draw_halvings(ax)
    for w in SG_WINDOWS:
        col, lw, al = SG_STYLE[w]
        h = w // 2                       # drop SG edge-extrapolation (mode='interp')
        ax.plot(day[h:-h], curves[w][h:-h], color=col, lw=lw, alpha=al, zorder=4,
                label=f'SG-{w}d filtered')

    # black dot with light edge where SG-365 meets each halving (no label)
    for xh in HALVINGS.values():
        ax.scatter(xh, np.interp(xh, day, curves[365]), s=55, color='black',
                   edgecolors=COLOR_TXT, linewidth=1.2, zorder=6)

    ax.set_xlim(day.min(), day.max())
    ax.set_ylim(*YLIM)
    ax.set_xlabel('Days since genesis block')
    ax.set_ylabel(r'$n_{HR}$')
    ax.grid(True, alpha=0.3)
    raw_handle = Line2D([0], [0], marker='o', color='none', markerfacecolor='#cfc4b0',
                        markeredgecolor='none', markersize=5, linestyle='none',
                        label=r'Raw $n_{HR}$')
    sg_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=[raw_handle] + sg_handles, loc='upper right',
              labelcolor=COLOR_TXT, fontsize=8)

    plt.suptitle('Bitcoin Hashrate: Power-Law Exponent Drift',
                 color=COLOR_TXT, fontsize=14, y=0.975, fontweight='bold')
    plt.figtext(0.5, 0.935,
                r'$n_{HR}(t)=\log(H_2/H_1)/\log(t_2/t_1)$' f'    N={len(day)}',
                ha='center', va='top', color=COLOR_SUB, fontsize=10)
    plt.subplots_adjust(top=0.90, bottom=0.09, left=0.08, right=0.97)
    return fig


def main():
    ap = argparse.ArgumentParser(
        description='Time-resolved drift of the daily hashrate exponent n_HR(t)')
    ap.add_argument('--no-plot', action='store_true', help='compute only, no figure')
    args = ap.parse_args()

    n, day = load_nHR()
    curves = sg_curves(n, SG_WINDOWS, POLYORDER)

    print(f"n_HR: N={len(n)}  median={np.median(n):.2f}  std={n.std():.1f}")
    for w in SG_WINDOWS:
        c = curves[w]
        print(f"SG {w:3d}d drift range: [{c.min():7.2f}, {c.max():7.2f}]")
    print(f"SG 365d at halvings: " +
          "  ".join(f"{nm}={np.interp(x, day, curves[365]):.1f}"
                    for nm, x in HALVINGS.items()))
    print(f"plot ylim (fixed): {YLIM}")

    if args.no_plot:
        return

    plt.style.use(STYLE_FILE)
    fig = make_plot(day, n, curves)
    fig.savefig(OUT_PNG, dpi=300, facecolor=fig.get_facecolor())
    print(f"saved {OUT_PNG}")
    plt.show()


if __name__ == '__main__':
    main()
