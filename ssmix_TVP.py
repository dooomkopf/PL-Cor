#!/usr/bin/env python3
"""ssmix_TVP.py -- M2 ORCHESTRATOR: time-varying coupling gamma(t) of price & hashrate.

Math:  ssmix_TVP_methods.py        Plots: ssmix_TVP_plot{1,2}.py
Idea / red thread (step by step):  ssmix_TVP_README.txt

Run:  python ssmix_TVP.py [--SG 365] [--poly 2] [--bwg 365] [--win 365]
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ssmix_TVP_methods import compute, STYLE_FILE, COLOR_TXT, SG_WIN, BWG, CYC
import ssmix_TVP_plot1
import ssmix_TVP_plot2


def print_table(R):
    g, gamma, zP, zH = R['g'], R['gamma'], R['zP'], R['zH']
    print("=============== COUPLING  price <-> hashrate ===============")
    print("  gamma_exp  = our slope n_H/n_P   (n-space, fluctuations)")
    print("  gamma_fa   = same Kalman, FREE intercept alpha(t) (Adriano check 13.07.)")
    print("  gamma_orig = H/P diagnostic slope; log is only the power-law transform")
    print("  corr       = correlation of the two exponent signals (what you SEE)")
    print(f"  GLOBAL:   gamma_exp = {R['g0']:+.2f}     gamma_fa = {R['g0_fa']:+.2f}     gamma_orig = {R['lev_g']:+.2f}")
    print("    cycle | gamma_exp | gamma_fa | gamma_orig |  corr")
    print("    ------+-----------+----------+------------+------")
    for c, (a, b) in CYC.items():
        m = (g >= a) & (g < b)
        if m.sum() > 30:
            ge = float(np.median(gamma[m])); cr = float(np.corrcoef(zP[m], zH[m])[0, 1])
            gf = float(np.median(R['gamma_fa'][m]))
            print(f"     '{c}  |   {ge:+5.2f}   |  {gf:+5.2f}  |    {R['lev'].get(c, float('nan')):+5.2f}   | {cr:+5.2f}")
    if 'mod_noise' in R:
        print("\n  mod RTS noise channel:")
        print("    R_t = MC var(H) + gamma^2 * MC var(P); Student-t is in the MC draw,")
        print("    no residual-dependent Student-t reweighting inside Kalman.")
        print("    cycle | sd_P_MC | sd_H_MC | sd_R_t")
        print("    ------+---------+---------+-------")
        mn = R['mod_noise']
        for c, (a, b) in CYC.items():
            m = (g >= a) & (g < b)
            if m.sum() > 30:
                sx = float(np.sqrt(np.nanmedian(mn['var_x'][m])))
                sy = float(np.sqrt(np.nanmedian(mn['var_y'][m])))
                sr = float(np.sqrt(np.nanmedian(mn['R_last'][m])))
                print(f"     '{c}  | {sx:7.4f} | {sy:7.4f} | {sr:6.4f}")
    print("===========================================================")


def main():
    ap = argparse.ArgumentParser(description='M2: time-varying coupling gamma(t)')
    ap.add_argument('--SG', type=int, default=SG_WIN)
    ap.add_argument('--bw', type=float, default=None, help='ssmix smoothing bw [days] (default SG/8)')
    ap.add_argument('--bwg', type=float, default=BWG, help='gamma_exp(t) smoothing bandwidth [days]')
    ap.add_argument('--poly', type=int, choices=[1, 2], default=2)
    ap.add_argument('--win', type=int, default=365, help='rolling window for integrated gamma_exp slope [days]')
    ap.add_argument('--bwl', type=float, default=20.0, help='hashrate-space Kalman bandwidth (panel b) [days]')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()
    bw = args.bw if args.bw is not None else args.SG / 8.0

    R = compute(args.SG, bw, args.bwg, args.poly, args.win, args.bwl)
    print_table(R)
    if args.no_plot:
        return

    plt.style.use(STYLE_FILE)
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(13, 6))
    ssmix_TVP_plot1.draw(axA, R)
    ssmix_TVP_plot2.draw(axB, R)
    axB.set_xlabel('Days since Genesis Block')
    plt.suptitle('Price-Hashrate Coupling Analysis',
                 color=COLOR_TXT, fontsize=14, y=0.975, fontweight='bold')
    plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.08, hspace=0.32)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ssmix_TVP.png')
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print(f"saved {out}")
    plt.show()


if __name__ == '__main__':
    main()
