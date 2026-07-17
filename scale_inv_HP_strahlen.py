#!/usr/bin/env python3
"""Ray separation of the hashrate S-curve tails in the pair cloud.

Extension of scale_inv_HP.py (identical pair sampling, seeds 0/1, same 3
panels, all original stdout output) with:

  * ALL panels share the panel-1 axes: xlim (-8, 8), ylim (-40, 40).
  * Panels 1 and 3 show two temporary rays ("S-tail cut", slope 33/3.1 =
    10.65 from SS (0.7, 0) -> SP (3.8, 33); both rays parallel-shifted
    inward by 0.25 -> x-intercepts at +-0.45):
    Selected HASHRATE pair sets (reported on stdout):
        pos. tail: y >= 0 and x > ray   (right of the positive ray)
        neg. tail: y <= 0 and x < ray   (left of the negative ray)
  * Without --pairing panels 2/3 show ONLY the both-in classes and the grey
    rest (original view + S-tail cuts) -- no markers.
  * Panels 2 (P) and 3 (H), ONLY with --pairing A-B (A, B in
    pre/13/17/21/25, e.g. 'pre-13', '13-17'; 'pre-X' = pre x ANY cycle;
    same-cycle combos excluded, they equal the std 'both in' classes):
    two-colour half-filled markers for pairs with the EARLIER endpoint in
    epoch A and the LATER one in epoch B, independent of the rays -- left
    half = cycle colour of the EARLIER endpoint, right half = LATER
    endpoint; grey half = pre-cycle phase (t < 1425 d, before the '13
    cycle). First, then every n-th pair (--every, default 20).
  * Panels 2 and 3, ONLY with --pairing pre-X: additional cloud class
    'pre x cycle' (one endpoint in the pre-cycle phase, the other in
    '13..'25) in petrol (COL_PREX), thinned to the first, then every n-th
    point (--every).
  * Legends of panels 2/3: the both-in cycle entries ALWAYS stay (origin
    plot); dynamic per --pairing are only the marker entries -- 'pre-cycle'
    (only if pre is involved), the split-marker example (colours follow the
    pairing) and petrol (only when drawn).
  * --tmin excludes data before a given day (e.g. 364 = pre-market cut).
"""
import os
import argparse
from collections import Counter
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression

from data_io import load_cm, positive, days_since_genesis, GENESIS
from dailynHR import DATA_FILE, STYLE_FILE, CYCLES, COLORS
from scale_inv_HP import (random_logratios, cycle_idx, draw_colored,
                          N_PAIRS, COL_P, COL_H, GREY)

COLOR_TXT = '#ecdcc4'
SS_X = 0.45              # ray x-intercept (0.7 parallel-shifted inward by 0.25)
SLOPE = 33.0 / 3.1       # ray slope, from original SS (0.7, 0) -> SP (3.8, 33)
COL_PREX = '#008B8B'     # petrol: 'pre x cycle' class (one endpoint pre, one in a cycle)
XLIM = (-8, 8)           # identical axes for ALL panels
YLIM = (-40, 40)
EVERY = 20               # panel 2/3: first, then every n-th marker-set pair

EPOCH_TOKENS = ['pre', '13', '17', '21', '25']       # cycle_idx -1, 0, 1, 2, 3
EPOCH_IDX = {tok: k - 1 for k, tok in enumerate(EPOCH_TOKENS)}
PAIRING_CHOICES = (['pre-pre'] +
                   [f'pre-{b}' for b in EPOCH_TOKENS[1:]] +      # pre-13 .. pre-25
                   ['pre-X'] +                                   # pre x ANY cycle
                   [f'{a}-{b}' for i, a in enumerate(EPOCH_TOKENS)
                    for b in EPOCH_TOKENS[i + 1:] if a != 'pre'])
# same-cycle combos '13-13' etc. excluded: they equal the std 'both in' classes


def to_date(t):
    return (GENESIS + timedelta(days=float(t))).isoformat()


def report(name, t1, t2):
    tmin, tmax = np.minimum(t1, t2), np.maximum(t1, t2)
    print(f"{name}: N = {len(t1)}")
    if len(t1) == 0:                                 # empty set (e.g. --tmin cut)
        return
    for lbl, tt in (('earlier endpoint', tmin), ('later endpoint', tmax)):
        q5, q50, q95 = np.percentile(tt, [5, 50, 95])
        print(f"  {lbl:>17}: median {to_date(q50)}   5-95% [{to_date(q5)} .. {to_date(q95)}]")


def combo_table(cmin, cmax, names):
    """Counts of (earlier-cycle x later-cycle) combinations, descending."""
    lbl = lambda k: 'pre' if k < 0 else names[k]
    cnt = Counter(zip(cmin, cmax))
    tot = len(cmin)
    for (a, b), n in cnt.most_common():
        print(f"    {lbl(a):>4} x {lbl(b):<4}  {n:6d}  ({100*n/tot:5.1f}%)")


def epoch_col(k):
    """cycle_idx -> colour; -1 (pre-cycle) -> grey."""
    return GREY if k < 0 else COLORS[k]


def epoch_lbl(tok):
    """--pairing token -> display form: digit tokens get the cycle apostrophe."""
    return f"'{tok}" if tok.isdigit() else tok


def pairing_mask(pairing, cmin, cmax):
    """(earlier, later) epoch mask for a --pairing token; 'pre-X' = pre x ANY cycle."""
    a, b = pairing.split('-')
    if b == 'X':
        return (cmin == -1) & (cmax >= 0)
    return (cmin == EPOCH_IDX[a]) & (cmax == EPOCH_IDX[b])


def draw_split_markers(ax, lt, lx, cmin, cmax, mask, every):
    """First, then every n-th pair of mask as a half-filled two-colour marker
    (left half = earlier endpoint epoch, right half = later). Returns count."""
    sub = np.zeros(len(lt), bool)
    sub[np.where(mask)[0][::every]] = True
    for ka in range(-1, len(COLORS)):
        for kb in range(-1, len(COLORS)):
            m = sub & (cmin == ka) & (cmax == kb)
            if m.any():
                ax.plot(lt[m], lx[m], ls='none', marker='o', fillstyle='left',
                        markerfacecolor=epoch_col(ka), markerfacecoloralt=epoch_col(kb),
                        markeredgecolor='#1a1a1a', markeredgewidth=0.3,
                        markersize=6, alpha=0.9)
    return int(sub.sum())


def draw_prex(ax, lt, lx, cmin, cmax, every):
    """'pre x cycle' cloud class, thinned to the first, then every n-th point."""
    idx = np.where((cmin == -1) & (cmax >= 0))[0][::every]
    ax.scatter(lt[idx], lx[idx], s=16, alpha=0.85, color=COL_PREX,
               edgecolors='none', label='pre x cycle')


def main():
    ap = argparse.ArgumentParser(description='ray separation of hashrate S-tails')
    ap.add_argument('--n', type=int, default=N_PAIRS, help='number of random pairs')
    ap.add_argument('--every', type=int, default=EVERY,
                    help='two-colour marker for the first, then every n-th pair '
                         'of the marker set')
    ap.add_argument('--pairing', choices=PAIRING_CHOICES, default=None,
                    help="marker set = pairs with earlier endpoint in epoch A, "
                         "later in B (e.g. pre-13, 13-17; pre-X = pre x any "
                         "cycle); without this option: no markers")
    ap.add_argument('--tmin', type=float, default=None,
                    help='exclude data before this day, e.g. 364 = pre-market '
                         'cut (default: full record)')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()
    if args.every < 1:
        ap.error('--every must be >= 1')
    if args.n < 1:
        ap.error('--n must be >= 1')

    d = load_cm(DATA_FILE)
    t_all = days_since_genesis(d['date']).astype(float)
    P, tP = positive(d['P'], t_all)
    H, tH = positive(d['H'], t_all)
    if args.tmin is not None:                      # pre-market data cut
        mP_, mH_ = tP >= args.tmin, tH >= args.tmin
        P, tP, H, tH = P[mP_], tP[mP_], H[mH_], tH[mH_]
        print(f"data cut: t >= {args.tmin:g} d   (N_P = {len(P)}, N_H = {len(H)})")

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

    # ---- ray selection of the hashrate S-tails ----
    print(f"\nS-tail cut (off-center ray): x-intercept +-{SS_X}, slope = {SLOPE:.2f}")
    pos = (lxH >= 0) & (ltH > SS_X + lxH / SLOPE)        # right of positive ray
    neg = (lxH <= 0) & (ltH < -SS_X + lxH / SLOPE)       # left of negative ray

    tminH, tmaxH = np.minimum(t1H, t2H), np.maximum(t1H, t2H)
    cminH, cmaxH = cycle_idx(tminH, cyc), cycle_idx(tmaxH, cyc)
    tminP, tmaxP = np.minimum(t1P, t2P), np.maximum(t1P, t2P)
    cminP, cmaxP = cycle_idx(tminP, cyc), cycle_idx(tmaxP, cyc)
    for name, m in (('pos. tail (right of ray)', pos),
                    ('neg. tail (left of ray)', neg)):
        report(name, t1H[m], t2H[m])
        print("  endpoint-cycle combos (earlier x later):")
        combo_table(cminH[m], cmaxH[m], names)

    # ---- marker sets for panels 2 (P) and 3 (H) ----
    posP = (lxP >= 0) & (ltP > SS_X + lxP / SLOPE)       # right of positive ray
    negP = (lxP <= 0) & (ltP < -SS_X + lxP / SLOPE)      # left of negative ray
    if args.pairing:
        markH = pairing_mask(args.pairing, cminH, cmaxH)
        markP = pairing_mask(args.pairing, cminP, cmaxP)
        for sym, mk, t1x, t2x in (('H', markH, t1H, t2H), ('P', markP, t1P, t2P)):
            if mk.any():
                report(f"{sym} pairing {args.pairing} (marker set, ray-independent)",
                       t1x[mk], t2x[mk])
            else:
                print(f"{sym} pairing {args.pairing}: N = 0")
    else:
        markH = pos | neg
        markP = posP | negP
    src = f"pairing {args.pairing}" if args.pairing else "beyond-ray"

    if args.no_plot:
        return

    plt.style.use(STYLE_FILE)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7.5))
    xx = np.linspace(-8, 8, 50)
    yy = np.array([0.0, YLIM[1]])
    ray_x = SS_X + yy / SLOPE

    # panel 1: no cycle colouring, plus the two rays
    ax1.scatter(ltP, lxP, s=4, alpha=0.18, color=COL_P, edgecolors='none', label='Price (BTC)')
    ax1.scatter(ltH, lxH, s=4, alpha=0.18, color=COL_H, edgecolors='none', label='Hashrate')
    ax1.plot(xx, bP * xx, color=COL_P, lw=2.0, label=f'Average Exponent  n = {bP:.2f}')
    ax1.set_title('BTC + Hashrate', color=COLOR_TXT, fontsize=12,
                  fontweight='bold', pad=10)

    # panel 2: price coloured by cycle (no fit line -- only in panel 1)
    draw_colored(ax2, ltP, lxP, sameP, c1P, GREY, names)
    if args.pairing == 'pre-X':
        draw_prex(ax2, ltP, lxP, cminP, cmaxP, args.every)
    if args.pairing:
        nP_mark = draw_split_markers(ax2, ltP, lxP, cminP, cmaxP, markP, args.every)
        print(f"panel 2: two-colour markers, first then every {args.every}th "
              f"{src} pair ({nP_mark} of {markP.sum()})")
    ax2.set_title('BTC Price Pairs Coloured by Cycle', color=COLOR_TXT, fontsize=12,
                  fontweight='bold', pad=10)

    # panel 3: hashrate coloured by cycle; with --pairing also markers/petrol
    draw_colored(ax3, ltH, lxH, sameH, c1H, GREY, names)
    if args.pairing == 'pre-X':
        draw_prex(ax3, ltH, lxH, cminH, cmaxH, args.every)
    if args.pairing:
        nH_mark = draw_split_markers(ax3, ltH, lxH, cminH, cmaxH, markH, args.every)
        print(f"panel 3: two-colour markers, first then every {args.every}th "
              f"{src} pair ({nH_mark} of {markH.sum()})")
    ax3.set_title('Hashrate Pairs Coloured by Cycle', color=COLOR_TXT, fontsize=12,
                  fontweight='bold', pad=10)

    for ax in (ax1, ax3):
        ax.plot(ray_x, yy, color='#9a9a9a', lw=0.9)
        ax.plot(-ray_x, -yy, color='#9a9a9a', lw=0.9)

    for ax in (ax1, ax2, ax3):
        ax.axhline(0, color='#6f5d46', lw=0.8)
        ax.axvline(0, color='#6f5d46', lw=0.8)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_xlabel(r'$\log(t_2/t_1)$')
        ax.set_ylabel(r'$\log(X_2/X_1)$')
        ax.grid(True, alpha=0.3, ls='--')

    h1 = [Line2D([0], [0], marker='o', color='none', markerfacecolor=COL_P,
                 markersize=5, label='BTC Price'),
          Line2D([0], [0], marker='o', color='none', markerfacecolor=COL_H,
                 markersize=5, label='Hashrate'),
          Line2D([0], [0], color=COL_P, lw=2.0, label=f'Average Exponent  n = {bP:.2f}'),
          Line2D([0], [0], color='#9a9a9a', lw=0.9, label='S-tail cut')]
    ax1.legend(handles=h1, loc='upper left', labelcolor=COLOR_TXT, fontsize=10)

    # legends of panels 2/3: both-in entries ALWAYS stay; dynamic per
    # --pairing are only the marker entries
    extra = []
    if args.pairing:
        pa, pb = args.pairing.split('-')
        mfc_l = epoch_col(EPOCH_IDX[pa])
        mfc_r = COL_PREX if pb == 'X' else epoch_col(EPOCH_IDX[pb])
        if 'pre' in (pa, pb):
            extra.append(Line2D([0], [0], marker='o', color='none',
                                markerfacecolor=GREY, markeredgecolor='#1a1a1a',
                                markeredgewidth=0.3, markersize=6, label='pre-cycle'))
        extra.append(Line2D([0], [0], ls='none', marker='o', fillstyle='left',
                            markerfacecolor=mfc_l, markerfacecoloralt=mfc_r,
                            markeredgecolor='#1a1a1a', markeredgewidth=0.3,
                            markersize=6,
                            label=('in pre AND any cycle' if pb == 'X' else
                                   f'in {epoch_lbl(pa)} AND {epoch_lbl(pb)}-cycle')))
    for ax in (ax2, ax3):
        hs = ax.get_legend_handles_labels()[0]
        for hnd in hs:
            if hnd.get_label().startswith('both in '):
                hnd.set_label(hnd.get_label() + '-cycle')
        ax.legend(handles=hs + extra, loc='upper left', labelcolor=COLOR_TXT,
                  fontsize=9)

    sup = 'Exponent Stability Test  Price vs Hashrate'
    if args.tmin is not None:
        sup += f'  (t ≥ {args.tmin:g} d)'
    plt.suptitle(sup, color=COLOR_TXT, fontsize=13, fontweight='bold', y=0.97)
    plt.subplots_adjust(top=0.88, bottom=0.10, left=0.04, right=0.99, wspace=0.18)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'scale_inv_HP_strahlen.png')
    fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    print(f"saved {out}")
    plt.show()


if __name__ == '__main__':
    main()
