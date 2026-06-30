#!/usr/bin/env python3
"""Daily power-law exponent of the Bitcoin HASHRATE (n_HR) -- DISPLAY ONLY.

Local power-law exponent of the hashrate H, sampled between consecutive days:
  n_HR = log(H_{t+1}/H_t) / log(t_{t+1}/t_t)     (local exponent of H ~ A t^n)
NO bin filter. Huber location as the robust peak estimator (best for the spike).

Also serves as a SHARED MODULE: daily_exponent(), DATA_FILE, STYLE_FILE, CYCLES,
COLORS are imported by the other PL-Cor analyses.

Deliberately NO distribution fit yet: we do not know whether n_HR is Laplace,
Student-t, something else, or has no stable center at all. So we only SHOW the
empirical histogram (log PDF) -- global and per halving cycle -- plus the Huber
peak as a marker. The fit/model decision comes later, from looking at this.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from data_io import load_cm, positive, days_since_genesis

# ------------------------------ parameters -------------------------------
HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(HERE, 'cm_data.csv')
STYLE_FILE = os.path.join(HERE, 'public.mplstyle')
HUBER_K    = 1.0                       # Huber tuning (1.0 for fat-tail robustness)
CYCLES     = {"'13": (1425, 2744), "'17": (2744, 4146),     # days since genesis
              "'21": (4146, 5586), "'25": (5586, 7044)}
COLORS     = ['#1f77ff', '#90EE90', '#FF69B4', '#FFD700']
N_BINS     = 100
XQUANT     = 99.0                      # central % of |n-median| that sets the x-range
# -------------------------------------------------------------------------


def huber_location(x, k=HUBER_K, tol=1e-6, max_iter=100):
    """Huber M-estimator of location (robust peak estimate for fat-tail data)."""
    x = np.asarray(x)
    mad = np.median(np.abs(x - np.median(x)))
    if mad == 0:
        mad = np.std(x)
    if mad == 0:
        return np.median(x)
    scale = mad / 0.6745
    mu = np.median(x)
    for _ in range(max_iter):
        r = (x - mu) / scale
        w = np.where(np.abs(r) <= k, 1.0, k / np.abs(r))
        mu_new = np.sum(w * x) / np.sum(w)
        if np.abs(mu_new - mu) < tol:
            break
        mu = mu_new
    return mu


def daily_exponent(X, t):
    """Local power-law exponent between consecutive days:
    n = log(X2/X1) / log(t2/t1). Returns (n, left-day)."""
    x1, x2, t1, t2 = X[:-1], X[1:], t[:-1], t[1:]
    ok = (x1 > 0) & (x2 > 0) & (t1 > 0) & (t2 > 0) & (t2 != t1)
    n = np.log(x2[ok] / x1[ok]) / np.log(t2[ok] / t1[ok])
    d = t1[ok]
    fin = np.isfinite(n)
    return n[fin], d[fin]


def hist_panel(ax, data, color, bins):
    """log-PDF scatter of one sample (raw distribution, nothing overlaid)."""
    ctr = (bins[:-1] + bins[1:]) / 2
    h, _ = np.histogram(data, bins=bins, density=True)
    cnt, _ = np.histogram(data, bins=bins)
    v = cnt > 0
    ax.scatter(ctr[v], h[v], s=24, alpha=0.6, c=color,
               edgecolors='#222222', linewidth=0.3)
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1e-1)


def main():
    ap = argparse.ArgumentParser(description='Daily hashrate exponent n_HR (display only)')
    ap.add_argument('--no-plot', action='store_true', help='skip the figures')
    args = ap.parse_args()

    d = load_cm(DATA_FILE)
    t_age = days_since_genesis(d['date'])
    H, t = positive(d['H'], t_age)
    n_HR, day = daily_exponent(H, t)

    lim = np.percentile(np.abs(n_HR - np.median(n_HR)), XQUANT)
    xlim = (-lim, lim)
    bins = np.linspace(*xlim, N_BINS + 1)

    print(f"daily n_HR (hashrate exponent):  N={len(n_HR)}  median={np.median(n_HR):.2f}")
    print(f"x-range (central {XQUANT:.0f}% of |n-median|): +/-{lim:.0f}")
    print("\nper-cycle counts (n_HR), display only:")
    print("  cyc     N")
    for name, (s, e) in CYCLES.items():
        cn = n_HR[(day >= s) & (day < e)]
        if len(cn) >= 10:
            print(f"  {name:4s} {len(cn):5d}")
    if args.no_plot:
        return

    plt.style.use(STYLE_FILE)

    # ---- Fenster 1: GLOBAL (alle Zyklen gefaerbt uebereinander) ----
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    for (name, (s, e)), col in zip(CYCLES.items(), COLORS):
        cn = n_HR[(day >= s) & (day < e)]
        if len(cn) < 10:
            continue
        hist_panel(ax1, cn, col, bins)
    ax1.set_xlim(*xlim)
    ax1.set_xlabel(r'daily $n_{HR} = \log(H_2/H_1)\,/\,\log(t_2/t_1)$')
    ax1.set_ylabel('PDF')
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8,
                      label=name) for name, c in zip(CYCLES, COLORS)]
    ax1.legend(handles=handles, loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, ls='--')
    plt.suptitle(r"BTC hashrate daily $n_{HR}$ -- global  [Huber peak, no fit]",
                 fontsize=13, y=0.975)
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(HERE, 'dailynHR_global.png'), dpi=300,
                facecolor=fig1.get_facecolor())

    # ---- Fenster 2: PRO ZYKLUS (2x2) ----
    fig2, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax2, (name, (s, e)), col in zip(axes.flatten(), CYCLES.items(), COLORS):
        cn = n_HR[(day >= s) & (day < e)]
        ax2.set_xlim(*xlim)
        ax2.set_xlabel(r'daily $n_{HR}$')
        ax2.set_ylabel('PDF')
        ax2.grid(True, alpha=0.3, ls='--')
        if len(cn) < 50:
            ax2.set_title(f"{name} cycle  (n={len(cn)})", fontsize=11)
            continue
        hist_panel(ax2, cn, col, bins)
        ax2.set_title(f"{name} cycle   (N={len(cn)})", fontsize=11)
    plt.suptitle(r"BTC hashrate daily $n_{HR}$ by halving cycle  [Huber peak, no fit]",
                 fontsize=14, y=0.985)
    plt.subplots_adjust(top=0.92, hspace=0.32, wspace=0.22)
    plt.savefig(os.path.join(HERE, 'dailynHR_cycles.png'), dpi=300,
                facecolor=fig2.get_facecolor())

    plt.show()


if __name__ == '__main__':
    main()
