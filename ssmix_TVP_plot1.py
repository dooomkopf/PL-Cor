#!/usr/bin/env python3
"""ssmix_TVP_plot1.py -- PLOT (a): OUR coupling in EXPONENT space, gamma_exp(t).

n_H = gamma_exp(t) * n_P, estimated by our robust per-cycle TVP Kalman (step 2).
This is the fluctuation coupling -- our method, in the stationary exponent space.
"""
from ssmix_TVP_methods import COL_G, COLOR_TXT, draw_halvings, legend

COL_FA = '#8fd694'   # gruen -- free-intercept check curve (gamma_fa, Adriano 13.07.)


def draw(ax, R):
    g, gamma, sg_ = R['g'], R['gamma'], R['sg_']
    ax.axhline(0.0, color='#6f5d46', lw=0.8)
    ax.fill_between(g, gamma - 2 * sg_, gamma + 2 * sg_, color=COL_G, alpha=0.30)
    ax.plot(g, gamma, color=COL_G, lw=1.6, label='γ_exp ±2σ')
    # free-intercept check: same Kalman, alpha(t) free instead of pinned to 0
    ax.fill_between(g, R['gamma_fa'] - 2 * R['sg_fa'], R['gamma_fa'] + 2 * R['sg_fa'],
                    color=COL_FA, alpha=0.15)
    ax.plot(g, R['gamma_fa'], color=COL_FA, lw=1.4, label='γ_exp free α(t) ±2σ')
    ax.set_ylabel('γ_exp')
    ax.set_title(f"(a) EXPONENT space:  n_H = γ_exp · n_P   (global {R['g0']:+.2f})",
                 color=COLOR_TXT, fontsize=12)
    draw_halvings(ax)
    legend(ax)
