#!/usr/bin/env python3
"""Distribution candidates for the daily hashrate exponent n_HR.

Fits Normal, Laplace, Student-t, Generalized-Normal (Subbotin) and Cauchy by
MLE and compares them PER halving cycle (and global) via log-likelihood,
AIC/BIC, KS (core) and Anderson-Darling (tails). The winner tells us which
noise law the hashrate exponent follows -- input for the noise model.

Since location AND scale drift per cycle (see dailynHR.py), the per-cycle fit
is the honest one; the global fit is shown only for contrast.

WORKFLOW / where the numbers go (provenance chain -- keep in sync!):
The winning per-cycle Student-t parameters (sigma, nu) printed by this script
are hardcoded as HASH_CYC in SSmix/ssmix_HP.py. The noise is MEASURED ONCE
here, in exponent space, and every downstream estimator (robust_smoother
band, ssmix_TVP: tvp_gamma R_t, level_tvp_kalman noise='ours') takes it as a
KNOWN, FIXED input -- it is never re-fitted in-model. Evidence plots:
dailynHR_distfit_cycles.png (the per-cycle fits) and
dailynHR_distfit_global.png (why one global scale would be wrong).
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from data_io import load_cm, positive, days_since_genesis
from dailynHR import (daily_exponent, huber_location, CYCLES, COLORS,
                      STYLE_FILE, XQUANT, DATA_FILE)

# candidate distributions (scipy) -- gennorm shape beta: 1=Laplace, 2=Normal
DISTS = {
    'Student-t':          stats.t,
    'Generalized Normal': stats.gennorm,   # beta interpolates Laplace(1)<->Normal(2)
}
LINESTYLE = {'Student-t': '--', 'Generalized Normal': '-.'}
# FIXED colour per distribution -- same mapping in every panel + legend
DIST_COLORS = {'Student-t': '#f5a623', 'Generalized Normal': '#3ec97a'}


def _ad(samples, cdf):
    """Anderson-Darling statistic (tail-sensitive)."""
    s = np.sort(samples)
    F = np.clip(cdf(s), 1e-12, 1 - 1e-12)
    n = len(s)
    i = np.arange(1, n + 1)
    return -n - np.sum((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1]))) / n


def fit_compare(data):
    """MLE-fit each candidate; return name -> metrics dict, sorted by AIC."""
    out = {}
    n = len(data)
    for name, dist in DISTS.items():
        try:
            p = dist.fit(data)
            ll = float(np.sum(dist.logpdf(data, *p)))
            k = len(p)
            ks = float(stats.kstest(data, dist.cdf, args=p).statistic)
            ad = _ad(data, lambda x: dist.cdf(x, *p))
            out[name] = dict(p=p, ll=ll, k=k, aic=2 * k - 2 * ll,
                             bic=k * np.log(n) - 2 * ll, ks=ks, ad=ad)
        except Exception:
            out[name] = dict(p=None, ll=np.nan, k=0, aic=np.inf, bic=np.inf,
                             ks=np.nan, ad=np.nan)
    return dict(sorted(out.items(), key=lambda kv: kv[1]['aic']))


def shape_note(name, p):
    """Human-readable shape parameter for the heavy-tail families."""
    if p is None:
        return "fit-fail"
    if name == 'Normal':
        return f"(μ={p[0]:.1f}, σ={p[1]:.0f})"
    if name == 'Student-t':            # scipy order (ν,μ,σ) -> show μ first
        return f"(μ={p[1]:.1f}, σ={p[2]:.0f}, ν={p[0]:.1f})"
    if name == 'Generalized Normal':   # scipy order (β,μ,α) -> show μ first
        return f"(μ={p[1]:.1f}, α={p[2]:.0f}, β={p[0]:.2f})"
    return ""


def print_table(label, res):
    print(f"\n=== {label} ===   (sorted by AIC; lower = better)")
    print(f"  {'dist':20s} {'dAIC':>8s} {'dBIC':>8s} {'KS':>7s} {'AD':>8s}  shape")
    aic0 = min(r['aic'] for r in res.values())
    bic0 = min(r['bic'] for r in res.values())
    for name, r in res.items():
        sh = shape_note(name, r['p']) if r['p'] is not None else "fit-fail"
        print(f"  {name:20s} {r['aic']-aic0:8.1f} {r['bic']-bic0:8.1f} "
              f"{r['ks']:7.3f} {r['ad']:8.1f}  {sh}")
    win = next(iter(res))
    head = min(res, key=lambda k: res[k]['ks'] if np.isfinite(res[k]['ks']) else np.inf)
    tail = min(res, key=lambda k: res[k]['ad'] if np.isfinite(res[k]['ad']) else np.inf)
    print(f"  -> winner AIC: {win}   head (KS): {head}   tail (AD): {tail}")


def _winners(res):
    """AIC (overall), head = min KS (center of mass), tail = min AD."""
    fin = lambda k, m: res[k][m] if np.isfinite(res[k][m]) else np.inf
    win = min(res, key=lambda k: fin(k, 'aic'))
    head = min(res, key=lambda k: fin(k, 'ks'))
    tail = min(res, key=lambda k: fin(k, 'ad'))
    return win, head, tail


def _panel(ax, name, data, x, bins, xlim):
    """One panel: log-PDF scatter + the candidate fits (fixed colour/order)."""
    ctr = (bins[:-1] + bins[1:]) / 2
    h, _ = np.histogram(data, bins=bins, density=True)
    cnt, _ = np.histogram(data, bins=bins)
    v = cnt > 0
    ax.scatter(ctr[v], h[v], s=18, alpha=0.5, c='#999999', edgecolors='none')
    res = fit_compare(data)
    for dname in DISTS:                            # FIXED order + colour every panel
        r = res[dname]
        if r['p'] is None:
            continue
        ax.plot(x, DISTS[dname].pdf(x, *r['p']), lw=1.7, ls=LINESTYLE[dname],
                color=DIST_COLORS[dname],
                label=f"{dname} {shape_note(dname, r['p'])}".strip())
    win, head, tail = _winners(res)
    ax.set_yscale('log')
    ax.set_xlim(*xlim)
    ax.set_ylim(1e-5, 1e-1)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel(r'daily $n_{HR}$')
    ax.set_ylabel('PDF')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, ls='--')
    info = ("best-fitting distribution:\n"
            f"  overall (AIC):       {win}\n"
            f"  center/peak (KS):    {head}\n"
            f"  tails (Anderson-D):  {tail}")
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=7.5, va='bottom',
            ha='left', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#1A1A1A',
                      edgecolor='#808080', alpha=0.85))


def plot_global(data, cyc_samples, xlim):
    """Fenster 1: global Laplace (location by Huber) as the compact global model
    -- peaked core + exponential tails. Behind it the 4 per-cycle Gaussians
    (labelled, so the cycles are identifiable) show that the global shape is a
    scale-mixture of cycle Gaussians, not an intrinsic fat tail."""
    plt.style.use(STYLE_FILE)
    x = np.linspace(*xlim, 800)
    bins = np.linspace(*xlim, 101)
    ctr = (bins[:-1] + bins[1:]) / 2
    N = sum(len(c) for c in cyc_samples.values())

    fig, ax = plt.subplots(figsize=(10, 7))
    h, _ = np.histogram(data, bins=bins, density=True)
    cnt, _ = np.histogram(data, bins=bins)
    v = cnt > 0
    ax.scatter(ctr[v], h[v], s=18, alpha=0.5, c='#999999', edgecolors='none')

    for (name, cn), col in zip(cyc_samples.items(), COLORS):     # cycle Gaussians
        mu, sg = stats.norm.fit(cn)
        w = len(cn) / N
        ax.plot(x, w * stats.norm.pdf(x, mu, sg), color=col, lw=1.2, ls='--',
                alpha=0.85, label=f"{name}:  μ={mu:5.1f}  σ={sg:4.0f}")

    mu_l = huber_location(data)                                  # global Laplace
    b_l = np.mean(np.abs(data - mu_l))
    ax.plot(x, stats.laplace.pdf(x, mu_l, b_l), color='#CCCCCC', lw=2.6,
            label=f"Laplace (Huber):  μ={mu_l:.1f}  b={b_l:.0f}")

    ax.set_yscale('log')
    ax.set_xlim(*xlim)
    ax.set_ylim(1e-5, 1e-2)
    ax.set_xlabel(r'daily $n_{HR}$')
    ax.set_ylabel('PDF')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, ls='--')
    plt.suptitle(r"BTC hashrate daily $n_{HR}$ -- global Laplace (Huber) over the 4 cycle Gaussians",
                 fontsize=12, y=0.975)
    plt.subplots_adjust(top=0.91)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'dailynHR_distfit_global.png'),
                dpi=300, facecolor=fig.get_facecolor())


def plot_cycles(cyc_samples, xlim):
    """Fenster 2: 2x2, one fit per halving cycle."""
    plt.style.use(STYLE_FILE)
    x = np.linspace(*xlim, 800)
    bins = np.linspace(*xlim, 101)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (name, data) in zip(axes.flatten(), cyc_samples.items()):
        _panel(ax, f"{name} cycle", data, x, bins, xlim)
    plt.suptitle(r"BTC hashrate daily $n_{HR}$ -- distribution fit per halving cycle",
                 fontsize=13, y=0.985)
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.22)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'dailynHR_distfit_cycles.png'),
                dpi=300, facecolor=fig.get_facecolor())


def main():
    ap = argparse.ArgumentParser(description='Distribution fits for n_HR')
    ap.add_argument('--no-plot', action='store_true', help='tables only')
    args = ap.parse_args()

    d = load_cm(DATA_FILE)
    t_age = days_since_genesis(d['date'])
    H, t = positive(d['H'], t_age)
    n_HR, day = daily_exponent(H, t)
    lim = np.percentile(np.abs(n_HR - np.median(n_HR)), XQUANT)

    cyc_samples = {}
    for name, (s, e) in CYCLES.items():
        cn = n_HR[(day >= s) & (day < e)]
        if len(cn) >= 50:
            cyc_samples[name] = cn

    for name, cn in cyc_samples.items():
        print_table(f"{name} cycle  (N={len(cn)})", fit_compare(cn))
    print_table(f"GLOBAL  (N={len(n_HR)})", fit_compare(n_HR))

    if not args.no_plot:
        plot_global(n_HR, cyc_samples, (-lim, lim))
        plot_cycles(cyc_samples, (-lim, lim))
        plt.show()


if __name__ == '__main__':
    main()
