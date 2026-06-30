#!/usr/bin/env python3
"""ssmix_TVP_plot2.py -- PLOT (b): coupling in ORIGINAL space, gamma_orig(t).

THREE estimates of the level-space gamma (ln H = a_t + gamma_t ln P):
  1. reference DOCX (gl_ref, weekly Gaussian RTS) -- FROZEN, matches the paper.
  2. our estimator + OUR noise (gl_ours, level Kalman, Student-t per cycle, option y).
  3. our exponent gamma integrated back (glev_from_exp).
"""
from ssmix_TVP_methods import COL_G, COL_REF, COLOR_TXT, draw_halvings, legend

COL_INT = '#c98fe0'   # our integrated curve


def draw(ax, R):
    g = R['g']
    ax.axhline(2.0, color='#888888', lw=1.1, ls='--', label='γ=2')
    ax.axhline(0.0, color='#6f5d46', lw=0.8)
    # Kalman RTS (Gaussian, weekly medians) -- FROZEN computation (reference_level_gamma)
    ax.plot(g, R['gl_ref'], color=COL_REF, lw=3.0, alpha=0.9, label='Kalman RTS')
    # mod RTS = our estimator (level Kalman with our noise) -- LILA, durchgezogen
    ax.plot(g, R['gl_ours'], color=COL_INT, lw=1.6, label='mod RTS')
    # integrated gamma_exp -- colour taken from PANEL 1 (COL_G), dashed
    ax.plot(g, R['glev_from_exp'], color=COL_G, lw=1.4, ls='--', label='integrated γ_exp')
    ax.set_ylabel('γ_orig')
    ax.set_title(f"(b) ORIGINAL space:  ln H = a + γ_orig · ln P   (static OLS {R['lev_g']:+.2f})",
                 color=COLOR_TXT, fontsize=12)
    draw_halvings(ax)
    legend(ax)
