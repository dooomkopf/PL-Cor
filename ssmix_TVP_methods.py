#!/usr/bin/env python3
"""ssmix_TVP_methods.py -- the MATH for M2 (time-varying coupling gamma(t)).

No plotting here. Every computation lives in this one module so the mathematics
stays separate and readable. The step-by-step idea: ssmix_TVP_README.txt.
"""
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.signal import savgol_filter
from data_io import load_cm, positive, days_since_genesis
from dailynHR import DATA_FILE
from ssmix_HP import (load_n, robust_smoother, price_cycle_b, hash_cycle_noise,
                      STYLE_FILE, HALVINGS, COLOR_TXT, SG_WIN)

# ------------------------------ parameters / style -----------------------
BWG     = 365            # default smoothing bandwidth for gamma_exp(t) [days]
REF_END_DAY = 6349   # DOCX sample end: 2026-05-23
REF_SIGMA_B = 0.015  # DOCX medium gamma random-walk specification
REF_SIGMA_A = 0.050  # reconstruction convention: loose intercept random walk
REF_R       = 0.500  # reconstruction convention: Gaussian level observation variance
COL_G   = '#6db3f2'      # OUR curves (exponent-space / integrated)
COL_REF = '#e8b84b'      # reference / reference curves
CYC = {'13': (1425, 2744), '17': (2744, 4146), '21': (4146, 5586), '25': (5586, 9999)}
# -------------------------------------------------------------------------


# ===== STEP 1: signals =====================================================
def load_signals(sg, bw, order):
    """SG-365 smooth EXPONENTS z_P,z_H (= the signal, fixed) + their per-cycle-noise
    uncertainties sz_P,sz_H, and the smoothed LOG-LEVELS sln_P,sln_H -- all on one grid."""
    gP, nP, sgP = load_n('P', sg)
    gH, nH, sgH = load_n('H', sg)
    rP = robust_smoother('P', gP, nP, sgP, bw, order)        # only for the uncertainty sz
    rH = robust_smoother('H', gH, nH, sgH, bw, order)
    lo, hi = max(int(gP[0]), int(gH[0])), min(int(gP[-1]), int(gH[-1]))
    g = np.arange(lo, hi + 1)
    zP = np.interp(g, gP, sgP); szP = np.interp(g, gP, rP['sz'])   # z = SG directly
    zH = np.interp(g, gH, sgH); szH = np.interp(g, gH, rH['sz'])
    d = load_cm(DATA_FILE); t = days_since_genesis(d['date'])
    P, tP = positive(d['P'], t); H, tH = positive(d['H'], t)
    w = sg + 1 if sg % 2 == 0 else sg
    slnP = savgol_filter(np.interp(g, tP, np.log(P)), w, 2)        # original/level space
    slnH = savgol_filter(np.interp(g, tH, np.log(H)), w, 2)
    return g, zP, szP, zH, szH, slnP, slnH


# ===== STEP 2: OUR coupling, exponent space (TVP errors-in-variables Kalman) =
def tvp_gamma(zP, szP, zH, szH, bwg, order, n_iter=15):
    """Errors-in-variables TVP Kalman regression  z_H = gamma_exp(t) * z_P + eps.
    gamma_exp(t) is an IRW latent (order 1/2); the per-cycle noise enters via the
    observation variance R(t) = sz_H^2 + gamma^2 sz_P^2 (Deming, time-varying)."""
    n = len(zH); d = order + 1
    F = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            F[i, j] = 1.0 / math.factorial(j - i)
    Id = np.eye(d)
    g0 = float(np.sum(zP * zH) / np.sum(zP * zP))            # global OLS slope (init)
    R_scale = float(np.median(szH ** 2 + g0 ** 2 * szP ** 2)) + 1e-9
    Qhi = R_scale / bwg ** (2 * (order + 1))
    Q = np.zeros((d, d)); Q[-1, -1] = Qhi
    gamma = np.full(n, g0)
    for _ in range(n_iter):
        Rt = szH ** 2 + gamma ** 2 * szP ** 2 + 1e-3 * R_scale
        a_pred = np.zeros((n, d)); P_pred = np.zeros((n, d, d))
        a_filt = np.zeros((n, d)); P_filt = np.zeros((n, d, d))
        a = np.zeros(d); a[0] = g0; P = np.eye(d) * 1e2
        for t in range(n):
            ap = F @ a; Pp = F @ P @ F.T + Q
            a_pred[t] = ap; P_pred[t] = Pp
            Ht = np.zeros((1, d)); Ht[0, 0] = zP[t]
            S = (Ht @ Pp @ Ht.T)[0, 0] + Rt[t]
            K = (Pp @ Ht.T).ravel() / S
            a = ap + K * (zH[t] - (Ht @ ap)[0])
            P = (Id - np.outer(K, Ht.ravel())) @ Pp
            a_filt[t] = a; P_filt[t] = P
        a_s = a_filt.copy(); P_s = P_filt.copy()
        for t in range(n - 2, -1, -1):
            C = P_filt[t] @ F.T @ np.linalg.solve(P_pred[t + 1], Id)
            a_s[t] = a_filt[t] + C @ (a_s[t + 1] - a_pred[t + 1])
            P_s[t] = P_filt[t] + C @ (P_s[t + 1] - P_pred[t + 1]) @ C.T
        gamma = a_s[:, 0]
    return gamma, np.sqrt(np.maximum(P_s[:, 0, 0], 0.0)), g0


# ===== STEP 3 + 4: original-space coupling, and OUR gamma integrated back =====
def rolling_slope(x, y, win):
    """rolling OLS slope of y on x (with intercept) + its standard error, centered."""
    n = len(x); sl = np.full(n, np.nan); se = np.full(n, np.nan); h = win // 2
    for t in range(h, n - h):
        xs = x[t - h:t + h]; ys = y[t - h:t + h]
        b, a = np.polyfit(xs, ys, 1)
        sxx = float(np.sum((xs - xs.mean()) ** 2))
        if sxx > 0:
            resid = ys - (a + b * xs)
            s2 = float(np.sum(resid ** 2)) / max(len(xs) - 2, 1)
            sl[t] = b; se[t] = np.sqrt(s2 / sxx)
    return sl, se


def integrate_exp_to_level(gamma, slnP, slnH):
    """STEP 4: reconstruct ln H from OUR exponent gamma by discrete integration
    d(ln H) = gamma_exp * d(ln P).  Returns the reconstructed ln H."""
    return slnH[0] + np.cumsum(gamma * np.gradient(slnP))


def level_gamma_per_cycle(slnP, slnH, g):
    """per-cycle OLS slope of ln H on ln P (descriptive, levels are I(1))."""
    out = {c: float(np.polyfit(slnP[(g >= a) & (g < b)], slnH[(g >= a) & (g < b)], 1)[0])
           for c, (a, b) in CYC.items() if ((g >= a) & (g < b)).sum() > 30}
    return out, float(np.polyfit(slnP, slnH, 1)[0])


def _percycle_scale(r, g):
    """robust per-cycle scale (1.4826*MAD) of residuals r, returned as a per-day array."""
    s = np.zeros(len(r))
    for c, (a, b) in CYC.items():
        m = (g >= a) & (g < b)
        if m.sum() > 10:
            s[m] = 1.4826 * float(np.median(np.abs(r[m] - np.median(r[m])))) + 1e-9
    s[s == 0] = (float(np.median(s[s > 0])) if (s > 0).any() else 1.0)
    return s


def _weekly_median(g, x, y, min_count=3):
    """Weekly medians, matching the reference reduced-frequency Kalman input."""
    week = ((g - g[0]) // 7).astype(int)
    gw, xw, yw = [], [], []
    for k in np.unique(week):
        m = week == k
        if m.sum() >= min_count:
            gw.append(float(np.median(g[m])))
            xw.append(float(np.median(x[m])))
            yw.append(float(np.median(y[m])))
    return np.asarray(gw), np.asarray(xw), np.asarray(yw)


def _rts_level_tvp(x, y, sigma_a, sigma_b, obs_var):
    """Gaussian level-space TVP Kalman + RTS smoother:
    ln H_t = a_t + gamma_t ln P_t + eps_t, with [a_t,gamma_t] random walk."""
    n = len(y)
    A = np.vstack([np.ones(n), x]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    F = np.eye(2)
    Q = np.diag([sigma_a ** 2, sigma_b ** 2])
    Id = np.eye(2)
    Rt = np.full(n, obs_var)

    a_pred = np.zeros((n, 2)); P_pred = np.zeros((n, 2, 2))
    a_filt = np.zeros((n, 2)); P_filt = np.zeros((n, 2, 2))
    st = coef.copy(); P = np.eye(2) * 1e2
    for i in range(n):
        ap = F @ st; Pp = F @ P @ F.T + Q
        a_pred[i] = ap; P_pred[i] = Pp
        Ht = np.array([[1.0, x[i]]])
        S = (Ht @ Pp @ Ht.T)[0, 0] + Rt[i]
        K = (Pp @ Ht.T).ravel() / S
        st = ap + K * (y[i] - (Ht @ ap)[0])
        P = (Id - np.outer(K, Ht.ravel())) @ Pp
        a_filt[i] = st; P_filt[i] = P

    a_sm = a_filt.copy(); P_s = P_filt.copy()
    for i in range(n - 2, -1, -1):
        C = P_filt[i] @ F.T @ np.linalg.solve(P_pred[i + 1], Id)
        a_sm[i] = a_filt[i] + C @ (a_sm[i + 1] - a_pred[i + 1])
        P_s[i] = P_filt[i] + C @ (P_s[i + 1] - P_pred[i + 1]) @ C.T
    return a_sm[:, 1], np.sqrt(np.maximum(P_s[:, 1, 1], 0.0))


def reference_level_gamma(g):
    """DOCX reconstruction of the reference yellow curve: raw weekly medians,
    Gaussian level-space TVP/RTS, medium sigma_b=0.015."""
    d = load_cm(DATA_FILE)
    age = days_since_genesis(d['date'])
    P, H, age = positive(d['P'], d['H'], age)
    m = age <= REF_END_DAY
    gw, xw, yw = _weekly_median(age[m], np.log(P[m]), np.log(H[m]))
    gamma_w, sd_w = _rts_level_tvp(xw, yw, REF_SIGMA_A, REF_SIGMA_B, REF_R)
    gamma = np.interp(g, gw, gamma_w); sd = np.interp(g, gw, sd_w)
    out_of_sample = (g < gw[0]) | (g > gw[-1])
    gamma[out_of_sample] = np.nan; sd[out_of_sample] = np.nan
    return gamma, sd


def level_tvp_kalman(slnP, slnH, g, bwl, noise, n_iter=6):
    """Reference-style level-space TVP Kalman:  ln H = a_t + gamma_t * ln P + eta,
    state [a_t, gamma_t] a random walk.   `noise` selects the observation model:
      noise='gauss' -> CONSTANT Gaussian eta            = the reference ORIGINAL (NOT robust).
      noise='ours'  -> Student-t per-cycle reweighting with OUR measured tail nu and a
                       robust per-cycle scale  = option y (our heavy-tailed noise model).
    Returns gamma(t) and its posterior standard deviation."""
    ours = (noise == 'ours')
    n = len(slnH)
    A = np.vstack([np.ones(n), slnP]).T
    coef = np.linalg.lstsq(A, slnH, rcond=None)[0]              # global [a0, gamma0]
    R0 = float(np.median((slnH - A @ coef) ** 2)) * 2.0 + 1e-9  # baseline obs variance
    Qa = R0 / bwl ** 2                                          # random-walk drift of a_t
    Qg = R0 / bwl ** 2 / (float(np.median(slnP ** 2)) + 1e-9)   # ... and of gamma_t (de-leveraged)
    F = np.eye(2); Q = np.diag([Qa, Qg]); Id = np.eye(2)
    if ours:                                                   # OUR MEASURED per-cycle noise -> level units
        t = g.astype(float)
        bP = price_cycle_b(g)                                  # Laplace scale, price exponent
        sH, nu = hash_cycle_noise(g)                           # Student-t (scale, tail), hashrate exponent
        var_x = (2.0 * bP ** 2) / t ** 2                       # price noise mapped to log-LEVEL (eps*dln t)
        var_y = (sH ** 2 * nu / np.maximum(nu - 2.0, 1.0)) / t ** 2   # hashrate noise -> level units
    a_state = np.full(n, coef[0]); gamma = np.full(n, coef[1]); P_s = np.zeros((n, 2, 2))
    for _ in range(n_iter):
        if ours:
            Rt = var_y + gamma ** 2 * var_x                    # OUR errors-in-variables (Deming) variance
            r = slnH - (a_state + gamma * slnP)
            Rt = Rt * (nu + r ** 2 / np.maximum(Rt, 1e-12)) / (nu + 1.0)   # OUR Student-t heavy tails
        else:
            Rt = np.full(n, R0)                                # Gaussian (reference)
        a_pred = np.zeros((n, 2)); P_pred = np.zeros((n, 2, 2))
        a_filt = np.zeros((n, 2)); P_filt = np.zeros((n, 2, 2))
        st = np.array([coef[0], coef[1]]); P = np.eye(2) * 1e2
        for t in range(n):
            ap = F @ st; Pp = F @ P @ F.T + Q
            a_pred[t] = ap; P_pred[t] = Pp
            Ht = np.array([[1.0, slnP[t]]])                    # time-varying regressor [1, ln P]
            S = (Ht @ Pp @ Ht.T)[0, 0] + Rt[t]
            K = (Pp @ Ht.T).ravel() / S
            st = ap + K * (slnH[t] - (Ht @ ap)[0])
            P = (Id - np.outer(K, Ht.ravel())) @ Pp
            a_filt[t] = st; P_filt[t] = P
        a_sm = a_filt.copy(); P_s = P_filt.copy()
        for t in range(n - 2, -1, -1):
            C = P_filt[t] @ F.T @ np.linalg.solve(P_pred[t + 1], Id)
            a_sm[t] = a_filt[t] + C @ (a_sm[t + 1] - a_pred[t + 1])
            P_s[t] = P_filt[t] + C @ (P_s[t + 1] - P_pred[t + 1]) @ C.T
        a_state = a_sm[:, 0]; gamma = a_sm[:, 1]
    return gamma, np.sqrt(np.maximum(P_s[:, 1, 1], 0.0))


# ===== orchestrated computation ============================================
def compute(SG, bw, bwg, poly, win, bwl):
    """run the whole M2 computation; return one results dict for the plots/print.
    `win`=rolling-slope window (integrated gamma_exp curve), `bwl`=level-Kalman bandwidth [days]."""
    g, zP, szP, zH, szH, slnP, slnH = load_signals(SG, bw, poly)
    gamma, sg_, g0 = tvp_gamma(zP, szP, zH, szH, bwg, poly)               # STEP 2 (our exponent gamma)
    lnH_rec = integrate_exp_to_level(gamma, slnP, slnH)                   # STEP 4: integrate it back
    glev_from_exp, _ = rolling_slope(slnP, lnH_rec, win)                  #  -> curve 1 (our integrated)
    gl_ref, gl_ref_sd = reference_level_gamma(g)                            # curve: reference DOCX (FROZEN)
    gl_ours, gl_ours_sd = level_tvp_kalman(slnP, slnH, g, bwl, noise='ours') # curve: option y (our noise)
    lev, lev_g = level_gamma_per_cycle(slnP, slnH, g)
    return dict(g=g, zP=zP, zH=zH, gamma=gamma, sg_=sg_, g0=g0,
                glev_from_exp=glev_from_exp, gl_ref=gl_ref, gl_ref_sd=gl_ref_sd,
                gl_ours=gl_ours, gl_ours_sd=gl_ours_sd,
                lev=lev, lev_g=lev_g)


def draw_halvings(ax):
    """halving lines H1-H4 with labels, into any axis."""
    y0, y1 = ax.get_ylim()
    for lab, dday in HALVINGS.items():
        ax.axvline(dday, color='#6f5d46', lw=0.8, ls='--', zorder=1)
        ax.text(dday, y1 - 0.06 * (y1 - y0), f' {lab}', color='#9b8a6f',
                fontsize=8, va='top', ha='left')


def legend(ax, loc='upper right'):
    ax.grid(True, alpha=0.3, ls='--')
    ax.legend(loc=loc, fontsize=8, facecolor='#1A1A1A',
              edgecolor='#808080', labelcolor='#E0E0E0')
