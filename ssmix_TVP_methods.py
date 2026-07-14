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
ADRIANO_END_DAY = 6349   # DOCX sample end: 2026-05-23
ADRIANO_SIGMA_B = 0.015  # DOCX medium gamma random-walk specification
ADRIANO_SIGMA_A = 0.050  # reconstruction convention: loose intercept random walk
ADRIANO_R       = 0.500  # reconstruction convention: Gaussian observation variance (hashrate space)
COL_G   = '#6db3f2'      # OUR curves (exponent-space / integrated)
COL_REF = '#e8b84b'      # Adriano / reference curves
CYC = {'13': (1425, 2744), '17': (2744, 4146), '21': (4146, 5586), '25': (5586, 9999)}
# -------------------------------------------------------------------------


# ===== STEP 1: signals =====================================================
def load_signals(sg, bw, order):
    """SG-365 smooth EXPONENTS z_P,z_H (= the signal, fixed) + their per-cycle-noise
    uncertainties sz_P,sz_H, and the smoothed sln_P,sln_H (price/hashrate space, log-transformed) -- all on one grid."""
    gP, nP, sgP = load_n('P', sg)
    gH, nH, sgH = load_n('H', sg)
    # sz below comes from the MEASURED, FIXED per-cycle noise channel
    # (PRICE_CYC/HASH_CYC in ssmix_HP.py; hashrate measured by
    # dailynHR_distfit.py) -- known input, never re-fitted here.
    rP = robust_smoother('P', gP, nP, sgP, bw, order)        # only for the uncertainty sz
    rH = robust_smoother('H', gH, nH, sgH, bw, order)
    lo, hi = max(int(gP[0]), int(gH[0])), min(int(gP[-1]), int(gH[-1]))
    g = np.arange(lo, hi + 1)
    zP = np.interp(g, gP, sgP); szP = np.interp(g, gP, rP['sz'])   # z = SG directly
    zH = np.interp(g, gH, sgH); szH = np.interp(g, gH, rH['sz'])
    d = load_cm(DATA_FILE); t = days_since_genesis(d['date'])
    P, tP = positive(d['P'], t); H, tH = positive(d['H'], t)
    w = sg + 1 if sg % 2 == 0 else sg
    slnP = savgol_filter(np.interp(g, tP, np.log(P)), w, 2)        # price space
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


# ===== STEP 2b: free-intercept check (Adriano, 13.07.2026) ===================
def tvp_gamma_free_alpha(zP, szP, zH, szH, bwg, order, n_iter=15):
    """Same errors-in-variables TVP Kalman as tvp_gamma, but with a FREE
    random-walk intercept alpha(t) instead of forcing z_H = gamma * z_P
    through the origin:
        z_H = alpha(t) + gamma_exp(t) * z_P + eps.
    alpha(t) can absorb autonomous hashrate-exponent drift (eg ASIC
    efficiency); if gamma shifts vs tvp_gamma, the pinned-origin gamma
    was biased (Adriano's hypothesis: underestimated)."""
    n = len(zH); d = order + 2                       # state = [alpha | gamma, derivs]
    F = np.eye(d)
    for i in range(1, d):                            # Taylor chain in the gamma block
        for j in range(i + 1, d):                    # only; alpha (row 0) = pure RW
            F[i, j] = 1.0 / math.factorial(j - i)
    Id = np.eye(d)
    A = np.vstack([np.ones(n), zP]).T
    a0, g0 = np.linalg.lstsq(A, zH, rcond=None)[0]   # global OLS WITH intercept (init)
    R_scale = float(np.median(szH ** 2 + g0 ** 2 * szP ** 2)) + 1e-9
    Q = np.zeros((d, d))
    Q[0, 0] = R_scale / bwg ** 2                     # alpha RW drifts on the bwg scale
    Q[-1, -1] = R_scale / bwg ** (2 * (order + 1))   # gamma IRW, exactly as in tvp_gamma
    gamma = np.full(n, g0)
    for _ in range(n_iter):
        Rt = szH ** 2 + gamma ** 2 * szP ** 2 + 1e-3 * R_scale
        a_pred = np.zeros((n, d)); P_pred = np.zeros((n, d, d))
        a_filt = np.zeros((n, d)); P_filt = np.zeros((n, d, d))
        a = np.zeros(d); a[0] = a0; a[1] = g0; P = np.eye(d) * 1e2
        for t in range(n):
            ap = F @ a; Pp = F @ P @ F.T + Q
            a_pred[t] = ap; P_pred[t] = Pp
            Ht = np.zeros((1, d)); Ht[0, 0] = 1.0; Ht[0, 1] = zP[t]
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
        gamma = a_s[:, 1]
    return gamma, np.sqrt(np.maximum(P_s[:, 1, 1], 0.0)), a_s[:, 0], float(g0)


# ===== STEP 3 + 4: hashrate-space coupling, and OUR gamma integrated back =====
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
    """per-cycle OLS slope of ln H on ln P (descriptive, the ln series are I(1))."""
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
    """Weekly medians, matching Adriano's reduced-frequency Kalman input."""
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
    """Gaussian hashrate-space TVP Kalman + RTS smoother:
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


def adriano_original_gamma(g):
    """DOCX reconstruction of Adriano's yellow curve: raw weekly medians,
    Gaussian hashrate-space TVP/RTS, medium sigma_b=0.015."""
    d = load_cm(DATA_FILE)
    age = days_since_genesis(d['date'])
    P, H, age = positive(d['P'], d['H'], age)
    m = age <= ADRIANO_END_DAY
    gw, xw, yw = _weekly_median(age[m], np.log(P[m]), np.log(H[m]))
    gamma_w, sd_w = _rts_level_tvp(xw, yw, ADRIANO_SIGMA_A, ADRIANO_SIGMA_B, ADRIANO_R)
    gamma = np.interp(g, gw, gamma_w); sd = np.interp(g, gw, sd_w)
    out_of_sample = (g < gw[0]) | (g > gw[-1])
    gamma[out_of_sample] = np.nan; sd[out_of_sample] = np.nan
    return gamma, sd


# ===== MC noise propagation into price space (template: MC/noise_movav.py) ====
MC_ANCHOR_WIN = 30     # centered moving-average anchor [days], as in the template
MC_N_PATHS    = 500    # MC paths, as in the template
MC_SEED       = 0      # fixed seed -> reproducible R_t


def _mc_obs_var(g, dist, scale, nu=None, seed=MC_SEED):
    """Observation variance R_t for mod RTS, extracted like noise_movav_H.py.

    noise_movav_H.py does not build one long random walk for the default band.
    For each anchor day it starts at the moving-average anchor, draws a drift-free
    daily exponent shock, maps it into the H/P path with

        d log(X) = n * d log(t),

    and reads the MC band at the next day. `log` is only the arithmetic transform
    for a multiplicative power-law step. The anchor cancels for the variance, so
    the correct Kalman input is the ensemble variance of those one-step
    transformed shocks, not the variance of a long path minus its own MA.
    """
    rng = np.random.default_rng(seed)
    g = np.asarray(g, dtype=float)
    scale = np.asarray(scale, dtype=float)
    dlnt = np.log((g + 1.0) / g)                       # same one-step factor as noise_movav_H
    n = len(g)
    if dist == 'student':
        steps = rng.standard_t(np.asarray(nu, dtype=float), size=(MC_N_PATHS, n)) * scale
    else:
        steps = rng.laplace(0.0, scale, size=(MC_N_PATHS, n))
    transformed_residual = steps * dlnt
    return np.var(transformed_residual, axis=0)


def level_tvp_kalman(slnP, slnH, g, bwl, noise, n_iter=6, return_noise=False):
    """Adriano-style hashrate-space TVP Kalman (ln H on ln P):  ln H = a_t + gamma_t * ln P + eta,
    state [a_t, gamma_t] a random walk.   `noise` selects the observation model:
      noise='gauss' -> CONSTANT Gaussian eta            = Adriano's ORIGINAL (NOT robust).
      noise='ours'  -> OUR measured per-cycle exponent noise, carried into price
                       space by MC integration around a moving-average anchor
                       (_mc_obs_var, template MC/noise_movav.py; completed
                       13.07.2026), combined Deming-style (var_y + gamma^2 var_x).
                       The Student-t shape is already in the MC draw; R_t stays
                       fixed by the measured noise channel, not by residuals.
    WORKFLOW: dailynHR_distfit.py measures the noise law per cycle -> fixed
    tables PRICE_CYC/HASH_CYC in ssmix_HP.py -> _mc_obs_var carries them into
    hashrate/price space (MC) -> R_t here. Never re-fitted in-model.
    Returns gamma(t) and its posterior standard deviation."""
    ours = (noise == 'ours')
    n = len(slnH)
    A = np.vstack([np.ones(n), slnP]).T
    coef = np.linalg.lstsq(A, slnH, rcond=None)[0]              # global [a0, gamma0]
    R0 = float(np.median((slnH - A @ coef) ** 2)) * 2.0 + 1e-9  # baseline obs variance
    Qa = R0 / bwl ** 2                                          # random-walk drift of a_t
    Qg = R0 / bwl ** 2 / (float(np.median(slnP ** 2)) + 1e-9)   # ... and of gamma_t (de-leveraged)
    F = np.eye(2); Q = np.diag([Qa, Qg]); Id = np.eye(2)
    if ours:                                                   # OUR MEASURED per-cycle noise -> hashrate/price space
        # noise = KNOWN, FIXED input, measured once in exponent space --
        # WORKFLOW block in ssmix_HP.py (HASH_CYC <- dailynHR_distfit.py)
        bP = price_cycle_b(g)                                  # Laplace scale, price exponent
        sH, nu = hash_cycle_noise(g)                           # Student-t (scale, tail), hashrate exponent
        var_x = _mc_obs_var(g, 'laplace', bP, seed=MC_SEED)        # price noise -> P transform
        var_y = _mc_obs_var(g, 'student', sH, nu, seed=MC_SEED + 1) # hashrate noise -> H transform
    a_state = np.full(n, coef[0]); gamma = np.full(n, coef[1]); P_s = np.zeros((n, 2, 2))
    for _ in range(n_iter):
        if ours:
            Rt = var_y + gamma ** 2 * var_x                    # OUR errors-in-variables (Deming) variance
        else:
            Rt = np.full(n, R0)                                # Gaussian (Adriano)
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
    Rt_final = var_y + gamma ** 2 * var_x if ours else np.full(n, R0)
    out = (gamma, np.sqrt(np.maximum(P_s[:, 1, 1], 0.0)))
    if return_noise and ours:
        info = dict(var_x=var_x, var_y=var_y, nu=nu, R_last=Rt_final)
        return out + (info,)
    return out


# ===== orchestrated computation ============================================
def compute(SG, bw, bwg, poly, win, bwl):
    """run the whole M2 computation; return one results dict for the plots/print.
    `win`=rolling-slope window (integrated gamma_exp curve), `bwl`=hashrate-space Kalman bandwidth [days]."""
    g, zP, szP, zH, szH, slnP, slnH = load_signals(SG, bw, poly)
    gamma, sg_, g0 = tvp_gamma(zP, szP, zH, szH, bwg, poly)               # STEP 2 (our exponent gamma)
    gamma_fa, sg_fa, alpha_fa, g0_fa = tvp_gamma_free_alpha(zP, szP, zH, szH, bwg, poly)  # STEP 2b: free-intercept check
    lnH_rec = integrate_exp_to_level(gamma, slnP, slnH)                   # STEP 4: integrate it back
    glev_from_exp, _ = rolling_slope(slnP, lnH_rec, win)                  #  -> curve 1 (our integrated)
    lnH_rec_fa = integrate_exp_to_level(gamma_fa, slnP, slnH)             # STEP 4 for gamma_fa (free-alpha check)
    glev_from_exp_fa, _ = rolling_slope(slnP, lnH_rec_fa, win)            #  -> its integrated curve
    gl_adr, gl_adr_sd = adriano_original_gamma(g)                            # curve: Adriano DOCX (FROZEN)
    gl_ours, gl_ours_sd, mod_noise = level_tvp_kalman(slnP, slnH, g, bwl, noise='ours',
                                                      return_noise=True) # curve: option y (our noise)
    lev, lev_g = level_gamma_per_cycle(slnP, slnH, g)
    return dict(g=g, zP=zP, zH=zH, gamma=gamma, sg_=sg_, g0=g0,
                gamma_fa=gamma_fa, sg_fa=sg_fa, alpha_fa=alpha_fa, g0_fa=g0_fa,
                glev_from_exp=glev_from_exp, glev_from_exp_fa=glev_from_exp_fa,
                gl_adr=gl_adr, gl_adr_sd=gl_adr_sd,
                gl_ours=gl_ours, gl_ours_sd=gl_ours_sd, mod_noise=mod_noise,
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
