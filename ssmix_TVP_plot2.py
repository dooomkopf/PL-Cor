#!/usr/bin/env python3
"""ssmix_TVP_plot2.py -- PLOT (b): coupling in HASHRATE space (ln H on ln P), gamma_orig(t).

FOUR estimates of the hashrate-space gamma (ln H = a_t + gamma_t ln P):
  1. Adriano DOCX (gl_adr, weekly Gaussian RTS) -- FROZEN, matches the paper.
  2. our estimator + OUR noise (gl_ours, hashrate-space Kalman, Student-t per cycle, option y).
  3. our exponent gamma integrated back (glev_from_exp).
  4. the free-intercept gamma_fa integrated back (glev_from_exp_fa; alpha(t) not integrated).
"""
from ssmix_TVP_methods import COL_G, COL_REF, COLOR_TXT, draw_halvings, legend

COL_INT = '#c98fe0'   # lila -- benutzt fuer 'mod RTS' (gl_ours)
COL_FA = '#8fd694'    # gruen -- same as PANEL 1 free-intercept curve (gamma_fa)


def draw(ax, R):
    g = R['g']
    ax.axhline(2.0, color='#888888', lw=1.1, ls='--', label='γ=2')
    ax.axhline(0.0, color='#6f5d46', lw=0.8)
    # Kalman RTS (Gaussian, weekly medians) -- FROZEN computation (adriano_original_gamma)
    ax.plot(g, R['gl_adr'], color=COL_REF, lw=3.0, alpha=0.9, label='Kalman RTS')
    # mod RTS = our estimator (hashrate-space Kalman with our noise) -- LILA, durchgezogen
    ax.plot(g, R['gl_ours'], color=COL_INT, lw=1.6, label='mod RTS')
    # integrated gamma_exp -- colour taken from PANEL 1 (COL_G), dashed
    ax.plot(g, R['glev_from_exp'], color=COL_G, lw=1.4, ls='--', label='integrated γ_exp')
    # integrated gamma_fa (free alpha) -- colour taken from PANEL 1 (COL_FA), dashed
    ax.plot(g, R['glev_from_exp_fa'], color=COL_FA, lw=1.4, ls='--',
            label='integrated γ_exp free α(t)')
    ax.set_ylabel('γ_orig')
    ax.set_title(f"(b) HASHRATE space:  ln H = a + γ_orig · ln P   (static OLS {R['lev_g']:+.2f})",
                 color=COLOR_TXT, fontsize=12)
    draw_halvings(ax)
    legend(ax)
