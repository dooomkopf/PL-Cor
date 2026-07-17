#!/usr/bin/env python3
"""Scale-invariance test for the BTC PRICE and HASHRATE power law.

Like scale_inv_with_gold.py, but: Hashrate instead of Gold, and random UNORDERED
pairs instead of incrementing a fixed delta-t. Draw N random pairs (t1, t2) -- t2
need NOT be later than t1, so log(t2/t1) can be negative. Plot

    y = log(X2/X1)   vs   x = log(t2/t1)

If X ~ t^n (scale-invariant), the cloud collapses onto a straight line through the
origin with slope n.

Three panels:
  1 -- no cycle colouring: just BTC price (gold) and hashrate (blue), full scale.
  2 -- PRICE pairs coloured by halving cycle (both endpoints same cycle), zoomed.
  3 -- HASHRATE pairs coloured by halving cycle, zoomed (same axes as panel 2).
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.linear_model import LinearRegression

from data_io import load_cm, positive, days_since_genesis
from dailynHR import DATA_FILE, STYLE_FILE, CYCLES, COLORS

N_PAIRS = 100000
COL_P = '#e8b84b'        # price (gold)
COL_H = '#6db3f2'        # hashrate (blue)
GREY = '#8a8a8a'         # mixed-cycle background in panels 2/3
COLOR_TXT = '#ecdcc4'
ZOOM_X = (-2, 2)
ZOOM_Y = (-15, 15)


def random_logratios(X, t, n_pairs, seed):
    """For n_pairs random UNORDERED index pairs return
    (log(t2/t1), log(X2/X1), t1, t2)."""
    rng = np.random.default_rng(seed)
    m = len(X)
    i = rng.integers(0, m, n_pairs)
    j = rng.integers(0, m, n_pairs)
    ok = (i != j) & (t[i] != t[j]) & (X[i] > 0) & (X[j] > 0)
    i, j = i[ok], j[ok]
    return np.log(t[j] / t[i]), np.log(X[j] / X[i]), t[i], t[j]


def cycle_idx(t, cycles):
    """cycle index 0..3 for each time, -1 if outside all cycles."""
    c = np.full(len(t), -1, int)
    for k, (s, e) in enumerate(cycles):
        c[(t >= s) & (t < e)] = k
    return c


def draw_colored(ax, lt, lx, same, c1, base_col, names):
    """Mixed-cycle pairs faint in base_col; same-cycle pairs in their cycle colour."""
    ax.scatter(lt[~same], lx[~same], s=4, alpha=0.10, color=base_col, edgecolors='none')
    for k, col in enumerate(COLORS):
        msk = same & (c1 == k)
        if msk.any():
            ax.scatter(lt[msk], lx[msk], s=16, alpha=0.85, color=col,
                       edgecolors='none', label=f"both in {names[k]}")


def main():
    ap = argparse.ArgumentParser(description='scale-invariance test: price & hashrate')
    ap.add_argument('--n', type=int, default=N_PAIRS, help='number of random pairs')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    d = load_cm(DATA_FILE)
    t_all = days_since_genesis(d['date']).astype(float)
    P, tP = positive(d['P'], t_all)
    H, tH = positive(d['H'], t_all)

    ltP, lxP, t1P, t2P = random_logratios(P, tP, args.n, seed=0)
    ltH, lxH, t1H, t2H = random_logratios(H, tH, args.n, seed=1)
    bP = LinearRegression(fit_intercept=False).fit(ltP.reshape(-1, 1), lxP).coef_[0]
    print(f"random unordered pairs: {args.n}")
    print(f"P : price fit slope (exponent) = {bP:6.2f}   pts={len(ltP)}")

    cyc = list(CYCLES.values())
    names = list(CYCLES.keys())
    c1H, c2H = cycle_idx(t1H, cyc), cycle_idx(t2H, cyc)
    c1P, c2P = cycle_idx(t1P, cyc), cycle_idx(t2P, cyc)
    sameH = (c1H == c2H) & (c1H >= 0)
    sameP = (c1P == c2P) & (c1P >= 0)
    print(f"H same-cycle pairs = {sameH.sum()} ({100*sameH.mean():.0f}%)")
    print(f"P same-cycle pairs = {sameP.sum()} ({100*sameP.mean():.0f}%)")

    print("\nper-cycle exponents n (fit through origin, same-cycle pairs):")
    print(f"{'cycle':>6}   {'n_P (price)':>12}   {'n_H (hashrate)':>14}")
    for k, name in enumerate(names):
        mP, mH = sameP & (c1P == k), sameH & (c1H == k)
        nP = LinearRegression(fit_intercept=False).fit(
            ltP[mP].reshape(-1, 1), lxP[mP]).coef_[0] if mP.any() else float('nan')
        nH = LinearRegression(fit_intercept=False).fit(
            ltH[mH].reshape(-1, 1), lxH[mH]).coef_[0] if mH.any() else float('nan')
        note = '   (cycle incomplete)' if name == "'25" else ''
        print(f"{name:>6}   {nP:12.2f}   {nH:14.2f}{note}")
    bH = LinearRegression(fit_intercept=False).fit(ltH.reshape(-1, 1), lxH).coef_[0]
    print(f"{'global':>6}   {bP:12.2f}   {bH:14.2f}   (all pairs)")

    if args.no_plot:
        return

    plt.style.use(STYLE_FILE)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7.5))
    xx = np.linspace(-8, 8, 50)

    # panel 1: no cycle colouring, full scale
    ax1.scatter(ltP, lxP, s=4, alpha=0.18, color=COL_P, edgecolors='none', label='Price (BTC)')
    ax1.scatter(ltH, lxH, s=4, alpha=0.18, color=COL_H, edgecolors='none', label='Hashrate')
    ax1.plot(xx, bP * xx, color=COL_P, lw=2.0, label=f'Average Exponent  n = {bP:.2f}')
    ax1.set_title('BTC + Hashrate', color=COLOR_TXT, fontsize=12,
                  fontweight='bold', pad=10)

    # panel 2: price coloured by cycle, zoomed (no fit line -- only in panel 1)
    draw_colored(ax2, ltP, lxP, sameP, c1P, GREY, names)
    ax2.set_xlim(*ZOOM_X)
    ax2.set_ylim(*ZOOM_Y)
    ax2.set_title('BTC Price Pairs Coloured by Cycle', color=COLOR_TXT, fontsize=12,
                  fontweight='bold', pad=10)

    # panel 3: hashrate coloured by cycle, zoomed
    draw_colored(ax3, ltH, lxH, sameH, c1H, GREY, names)
    ax3.set_xlim(*ZOOM_X)
    ax3.set_ylim(*ZOOM_Y)
    ax3.set_title('Hashrate Pairs Coloured by Cycle', color=COLOR_TXT, fontsize=12,
                  fontweight='bold', pad=10)

    for ax in (ax1, ax2, ax3):
        ax.axhline(0, color='#6f5d46', lw=0.8)
        ax.axvline(0, color='#6f5d46', lw=0.8)
        ax.set_xlabel(r'$\log(t_2/t_1)$')
        ax.set_ylabel(r'$\log(X_2/X_1)$')
        ax.grid(True, alpha=0.3, ls='--')

    from matplotlib.lines import Line2D
    h1 = [Line2D([0], [0], marker='o', color='none', markerfacecolor=COL_P,
                 markersize=10, label='BTC price (gold)'),
          Line2D([0], [0], marker='o', color='none', markerfacecolor=COL_H,
                 markersize=10, label='Hashrate (blue)'),
          Line2D([0], [0], color=COL_P, lw=2.0, label=f'Average Exponent  n = {bP:.2f}')]
    ax1.legend(handles=h1, loc='upper left', labelcolor=COLOR_TXT, fontsize=10)
    ax2.legend(loc='upper left', labelcolor=COLOR_TXT, fontsize=9)
    ax3.legend(loc='upper left', labelcolor=COLOR_TXT, fontsize=9)

    plt.suptitle('Exponent Stability Test  Price vs Hashrate',
                 color=COLOR_TXT, fontsize=13, fontweight='bold', y=0.97)
    plt.subplots_adjust(top=0.88, bottom=0.10, left=0.04, right=0.99, wspace=0.18)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scale_inv_HP.png')
    fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    print(f"saved {out}")
    plt.show()


if __name__ == '__main__':
    main()
