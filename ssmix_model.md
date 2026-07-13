# ssmix — Scale-Mixture State-Space model (exponent space, robust)

**Name:** `ssmix` = **S**tate-**S**pace + scale-**MIX**ture. The name is accurate
in two senses at once: (1) the heavy-tailed Laplace / Student-t observation noise
is represented as a **scale mixture of normals** (the trick that keeps it a Kalman
problem); (2) the two series use **mixed** observation laws — Laplace for the
price, Student-t for the hashrate. Script: `ssmix_HP.py` (**H**ashrate & **P**rice).
One-line pitch: *the correct, non-Gaussian replacement for a Gaussian Kalman
filter, posed in the stationary exponent space, with the observation noise law
taken from our empirical per-cycle noise analysis instead of assumed Gaussian.*

**Context:** this is **fundamental research** — the goal is to *understand* the
price–hashrate system and *exhaust* the model, not to produce a publication. An
honest "the coupling is at the data-resolution limit" is itself a result worth
knowing.

---

## ★ CURRENT STATE & DECISIONS — read this FIRST, keep it current

**GOAL (in order, all CORE — none is throwaway "scaffolding"):**
1. **ssmix marginal robust model** — a robust non-Gaussian state-space model of
   EACH exponent (n_P, n_H) using OUR MEASURED noise (Laplace b_P≈102 for price,
   Student-t per cycle for hashrate, §3b). Output: robust latent signal z ± band,
   velocity v, per-point outlier weights.
2. **M2** — the time-varying coupling exponent γ(t) (the directed price→hashrate
   question). `ssmix_TVP.py`.
3. **M3** — bivariate coupled model (data-limited).

**THE ORIGINAL IDEA (how we started — get this right):**
Build an **adapted smoother** that uses the MEASURED per-cycle noise as its
observation **noise channel**, and **estimate the signal** with it. Then **VERIFY,
per cycle, that the estimated signal is consistent with SG-365** (= our reference
signal). It is a **consistency check of the signal + noise decomposition**, done
per cycle — NOT a robust replacement of the signal.

- Reference signal: `SG-365(n_X)` — what the estimate should reproduce.
- The smoother **estimates** z using the noise channel (Laplace / Student-t, FIXED
  to the measured per-cycle width); z is **compared** to SG, not forced to it.
- If z ≈ SG per cycle **and** the residual matches the measured law → the ideas hold.
- (My error: I called the estimated z "the new robust signal", let it drift, used a
  GLOBAL scale, and babbled about Gauss. z is an ESTIMATE we CHECK against SG.)

**NOISE — MEASURED per-cycle (§3b); this IS the smoother's noise channel.**
**TWO tables — one per signal:**
- Price `n_P`: **Laplace per cycle (μ, b)**:
  '13 (5.4, 57) · '17 (8.7, 96) · '21 (7.4, 107) · '25 (−2.5, 103).
- Hashrate `n_H`: **Student-t per cycle (μ, s, ν)**:
  '13 (14.9, 219, 17.5) · '17 (−1.3, 377, 9.2) · '21 (3.2, 566, 21.8) · '25 (3.9, 697, 158.8).
- Use the per-cycle **width** as the FIXED noise scale (NOT a global value).
  The fitted offset/location μ is NOT fed into the Kalman noise channel. For the
  downstream test the noise is drift-free: draw around zero with the measured
  width and shape. Same channel feeds the band and M2.

**M2 / mod RTS — concrete current model (13.07.2026):**
- Main test stays in `n_P/n_H`: `n_H(t) = γ(t) n_P(t) + ε_t`, plus the already
  included free-intercept check `n_H(t) = α(t) + γ(t)n_P(t) + ε_t`.
- SG-365 is the reference curve. A freely estimated `z` is allowed as an add-on
  consistency check, but it is not allowed to redefine the idea.
- `mod RTS` is only the H/P diagnostic: `log` is used as a power-law arithmetic
  transform, because `d log(X) = n_X d log(t)`. No separate log-analysis space is
  introduced.
- The `mod RTS` noise width is extracted from the same MC mechanism as
  `noise_movav_H.py`: for each day, draw drift-free daily exponent noise from the
  measured per-cycle law, map it one step into H/P via `d log(X)=n d log(t)`, and
  take the ensemble variance of that transformed one-step residual.
- Therefore `R_t = Var_MC(H_noise) + γ(t)^2 Var_MC(P_noise)`. The Student-t shape
  is already present in the MC draw from `dailynHR_distfit.py`; there is no extra
  residual-dependent Student-t reweighting inside `mod RTS`.

**ERRORS I MADE (do not repeat):**
- treated the freely estimated `z` as a **replacement** for SG. Wrong framing:
  SG is the reference signal; free `z` is only a small diagnostic/add-on.
- made `--obs robust` the silent default.
- used ν=15 + a **global** MAD-s for hashrate instead of the **per-cycle μ + width** (§3b).

> ⚠️ §2–§5 below describe a robust-Kalman that *re-estimates* z — **SUPERSEDED** by
> the above where they imply `z` replaces SG. Keep free `z` only as a diagnostic.
> The Laplace/Student-t widths are the measured noise channel for bands and M2.

---

### Standard name in mathematics
- Overall: a **non-Gaussian / robust state-space model** = **robust Dynamic Linear
  Model (DLM)**; in the time-series literature a **structural time-series /
  unobserved-components model** with heavy-tailed observation noise.
- Latent `(z, v)` trend: a **local linear trend model**.
- Laplace/Student-t via scale mixture + EM: a **scale mixture of normals (SMN)**
  representation; estimation = **EM / IRLS robust Kalman smoother**.
- Canonical refs: Durbin & Koopman, *Time Series Analysis by State Space Methods*;
  Harvey, *Structural Time Series Models and the Kalman Filter*; West & Harrison,
  *Bayesian Forecasting and Dynamic Models* (the Student-t DLM).

---

## 0. Why (motivation)

- A Gaussian Kalman filter (e.g. a level-space `γ(t)` random-walk model) assumes
  **Gaussian** measurement noise. We have **measured** the daily-exponent noise and
  it is **not** Gaussian:
  - **Price** `n_P`: noise ≈ **Laplace** — peaked centre, exponential tails,
    near-invariant width `b_P ≈ 100` across halving cycles.
  - **Hashrate** `n_HR`: noise ≈ **Student-t** — heavier tails, stronger outliers,
    plausibly **regime-switching** scale and/or degrees of freedom.
- This asymmetry is **already encoded in our two noise scripts**:
  `noise_movav.py` (price) defaults to a **Laplace** cone; `noise_movav_H.py`
  (hashrate) defaults to a **Student-t** band. `ssmix` simply turns those
  empirical laws into the **observation model** of a proper filter.
- We work in **exponent space** (`n_P`, `n_HR`), which is stationary (I(0)),
  avoiding the spurious-regression danger of level space.

---

## 1. Notation

| symbol | meaning |
|---|---|
| `t` | integer day on the common grid (days since genesis) |
| `X ∈ {P, H}` | series: price `P`, hashrate `H` |
| `n^X_t` | local power-law exponent (observed) |
| `z^X_t` | latent **signal** (true slow exponent) |
| `v^X_t` | latent **velocity** (rate of change of the exponent) |
| `ε^X_t` | daily observation noise |

Observed local exponent (discrete, natural log):

$$
n^X_t \;=\; \frac{\ln\!\big(X_{t+1}/X_t\big)}{\ln\!\big(t_{t+1}/t_t\big)} .
$$

---

## 2. State equation (latent dynamics) — local linear trend / integrated random walk

$$
z_t = z_{t-1} + v_{t-1} + u_t, \qquad u_t \sim \mathcal N(0,\,Q_z),
$$
$$
v_t = v_{t-1} + w_t, \qquad\qquad\;\; w_t \sim \mathcal N(0,\,Q_v).
$$

Matrix form with state $a_t=(z_t,\,v_t)^\top$:

$$
a_t = F\,a_{t-1} + \eta_t,\quad
F=\begin{pmatrix}1&1\\[2pt]0&1\end{pmatrix},\quad
\eta_t\sim\mathcal N\!\Big(0,\;\mathrm{diag}(Q_z,Q_v)\Big).
$$

This is the standard **integrated-random-walk (IRW) smoother**. Its posterior mean
for `z` is a smooth curve — in the Gaussian case it is essentially the
**Whittaker–Henderson / Hodrick–Prescott** smoother, i.e. **the model-based twin of
SG-365**, but additionally with an **uncertainty band** and an explicit **velocity**.

---

## 3. Observation equation (the key part — non-Gaussian, empirical)

$$
n^X_t = z^X_t + \varepsilon^X_t .
$$

**Price (Laplace, near-invariant):**
$$
\varepsilon^P_t \sim \mathrm{Laplace}(0,\,b_P),\qquad
p(\varepsilon)=\frac{1}{2b_P}\exp\!\Big(-\frac{|\varepsilon|}{b_P}\Big),
\qquad b_P \approx 100 .
$$

**Hashrate (Student-t, heavier / regime-switching):**
$$
\varepsilon^H_t \sim t_{\nu_H}\!\big(0,\,s_H(t)\big),\qquad
p(\varepsilon)\;\propto\;\Big(1+\frac{\varepsilon^2}{\nu_H\,s_H^2}\Big)^{-\frac{\nu_H+1}{2}},
$$
with potentially time-varying scale / df: $s_H=s_H(t),\ \nu_H=\nu_H(t)$
(but see §6 — do **not** free-estimate everything).

---

## 3b. Calibrated noise parameters (MEASURED — these feed the observation model)

These are the actual fitted values from our existing noise scripts, so we never
have to re-gather them.

**Price `n_P` — Laplace per cycle** · Huber location μ, `b = mean|n − μ|`
(method of `/home/hz/Data/dailyn-fit-nobinfilter-huber.py`, computed per cycle):

| cycle | N | μ (Huber) | b (Laplace scale) |
|---|---|---|---|
| '13 | 1319 | 5.4 | 57 |
| '17 | 1402 | 8.7 | 96 |
| '21 | 1440 | 7.4 | 107 |
| '25 | 797 | −2.5 | 103 |

- b rises '13→'21, then ~flat. The '17+'21 combined fit gives b_P ≈ 102 (consistent).
  Auxiliary Sign-OU on '17+'21: κ̂ ≈ 102.9, σ̂ ≈ 197.5.

**Hashrate `n_HR` — Student-t per cycle** · source
`/home/hz/Data/zeitgeist/dailynHR_distfit.py` (AIC winner per cycle):

| cycle | N | μ | σ (scale) | ν (df) |
|---|---|---|---|---|
| '13 | 1319 | 14.9 | 219 | 17.5 |
| '17 | 1402 | −1.3 | 377 | 9.2 |
| '21 | 1440 | 3.2 | 566 | 21.8 |
| '25 | 797 | 3.9 | 697 | 158.8 |

- Student-t wins per cycle (AIC). The **global** pooled fit is Generalized Normal
  (β = 0.81, even fatter — an artefact of pooling rising per-cycle scales).
- → **s_H(t) is clearly time-varying (219 → 697, ×3.2 over the cycles); ν_H varies
  9–159** (fix it constant for identifiability, e.g. ν ≈ 15).

**The asymmetry (this is the empirical result):** price noise scale ≈ **stable**
(b_P ≈ 102); hashrate noise scale **grows strongly** (219 → 697) and is fat-tailed
— exactly the Laplace(stable) vs Student-t(time-varying) split ssmix assumes.

*(The daily-exponent noise scales are ~10²–10³ because n = Δln X / Δln t and
Δln t ≈ 1/t is tiny, so the exponent and its noise are numerically large.)*

---

## 4. Why this is tractable: Gaussian scale mixtures

Both Laplace and Student-t are **Gaussian scale mixtures**: conditional on a latent
per-point scale, the model is **linear-Gaussian**, so the **exact Kalman / RTS
smoother** applies and inference is an **EM / IRLS** loop — **no particle filter**.

**Student-t** as a scale mixture (the clean case):
$$
\varepsilon_t \mid w_t \sim \mathcal N\!\Big(0,\ \tfrac{s^2}{w_t}\Big),
\qquad w_t \sim \mathrm{Gamma}\!\Big(\tfrac{\nu}{2},\tfrac{\nu}{2}\Big)
\;\Longrightarrow\; \varepsilon_t \sim t_\nu(0,s).
$$

**Laplace** as a scale mixture:
$$
\varepsilon_t \mid \tau_t^2 \sim \mathcal N(0,\ \tau_t^2),
\qquad \tau_t^2 \sim \mathrm{Exponential}\!\Big(\tfrac{1}{2b^2}\Big)
\;\Longrightarrow\; \varepsilon_t \sim \mathrm{Laplace}(0,b).
$$

So given the latent scales, each observation is Gaussian with a **per-point
variance**; the fat tails enter **only** through that variance.

---

## 5. Inference — robust Kalman smoother via EM (IRLS)

Let $r_t = n_t - z_t$ be the current smoothed residual.

**E-step (re-weight):** compute the expected inverse-scale (the robustness weight):

- Student-t:
$$
\bar w_t \;=\; \mathbb E[w_t \mid r_t] \;=\; \frac{\nu+1}{\,\nu + r_t^2/s^2\,}.
$$
Large residual ⇒ small weight ⇒ the outlier is **down-weighted** (classic robust
Kalman). As $\nu\to\infty$ this $\to 1$ and the filter reduces to the Gaussian Kalman.

- Laplace: $\;\bar w_t \propto 1/|r_t|\;$ (an $L_1$ / least-absolute-deviations style
  weight).

**M-step (weighted Kalman):** run an **exact Kalman filter + RTS smoother** with a
**heteroscedastic** observation variance
$$
R_t = \frac{s^2}{\bar w_t}\quad(\text{Student-t}),\qquad
R_t = \frac{1}{\bar w_t}\ \text{-type}\quad(\text{Laplace}),
$$
update the states $(z_t,v_t)$, then update the parameters
$(b_P,\ s_H,\ \nu_H,\ Q_z,\ Q_v)$ by their **weighted MLE**.

**Iterate** E/M to convergence. Exact-Kalman-inside-EM ⇒ stable and fast.

---

## 6. Identifiability — the real constraint (regularization)

Free-estimating $s_H(t)$ **and** $\nu_H(t)$ **and** the process noise $(Q_z,Q_v)$
**simultaneously** is **not robustly identifiable**: observation noise and process
noise both produce short-term deviations and are only **weakly separable** without
repeated measurements or strong priors.

**Pragmatic, identifiable specification:**

- **Price:** Laplace with **global** $b_P$ (or weakly cycle-varying).
- **Hashrate:** Student-t with
  - $\nu_H$ **constant** (fixed, or a tight prior) — do **not** let df be time-varying;
  - $s_H(t)$ a **slow random walk on the log scale**,
    $\;\log s_H(t)=\log s_H(t-1)+\eta_t,\ \eta_t\ \text{small}$,
    **or** simply **fixed exogenously** from our per-cycle noise fits.
- **Process noise $Q$** strongly **regularized or fixed**; constrain the ratio
  $Q_z/Q_v$.
- Choose hyperparameters by **predictive likelihood / cross-validation /
  simulation-based calibration**, not by free ML.

---

## 7. Output (what ssmix delivers)

- $z^X_t \pm \sigma_t$: the latent exponent **signal with an uncertainty band** —
  better than SG-365 (model-based, robust, uncertainty-quantified).
- $v^X_t$: the **velocity** (rate of change of the exponent) with uncertainty.
- Per-point **outlier weights** $\bar w_t$: a diagnostic of *where* the fat-tail
  events sit.
- The fitted noise parameters $(b_P,\ s_H(t),\ \nu_H)$ — which **confirm the
  price/hashrate asymmetry as a result**, not an assumption.

---

## 8. Scope — honest (marginal only, NO coupling gain)

`ssmix` as above is a **per-series (marginal)** model. It produces better
**individual** signals $z_P, z_H$, but it says **nothing about price–hashrate
coupling**. It is a **better pre-filter** for any downstream coupling analysis,
**not** a coupling test.

---

## 9. Coupling extension (Option B) — bivariate **coupled** state-space

To extract coupling information, the two latent trends must share structure. Add
one or more of:

- **Common cycle factor** $c_t$ driving both:
  $$
  z^P_t = \alpha_P + \beta_P\, c_t + \dots,\qquad
  z^H_t = \alpha_H + \beta_H\, c_t + \dots
  $$
- **Cross-lag** coupling: $z^P_t$ depends on $z^H_{t-\ell}$ (or on the velocities):
  $$
  v^P_t = \phi\, v^P_{t-1} + \psi\, v^H_{t-\ell} + w^P_t .
  $$
- **Correlated process shocks**: $\mathrm{Cov}(u^P_t,\,u^H_t)\neq 0$.
- **Error-correction** (if a long-run relation $z^H \approx \gamma z^P$ holds):
  $$
  \Delta z^H_t = \dots - \kappa\big(z^H_{t-1} - \gamma\, z^P_{t-1}\big) + \dots
  $$

This is **where coupling lives** — and exactly the **data-limited, fragile** regime
(~4 halving cycles). It is the only version that could add genuine coupling
evidence, at the known risk of over-fitting.

---

## 10. Positioning (paper)

- **As a main contribution:** too much elegance for too little new insight.
- **As a robustness / methods appendix:** strong — a clean defense against
  *"a Gaussian Kalman filter is the wrong model"*.
- **The valuable result:** build the **conservative** variant (§6: $\nu_H$ constant,
  $s_H(t)$ slow/exogenous, $Q$ fixed), compare $z$, residuals, outlier weights and
  any coupling metric against SG-365. **If the conclusions stay stable, SG-365 is
  shown to be not merely convenient but robust against a formally better
  non-Gaussian model.**

---

## 11. Relation to existing code

- `noise_movav.py` (price) → **Laplace** default; `noise_movav_H.py` (hashrate) →
  **Student-t** default. The asymmetry of §3 is already empirically in place.
- SG-365 ≈ the Gaussian-case posterior mean of `z` under the IRW state model
  (Whittaker/HP equivalence). `ssmix` is the **robust, uncertainty-quantified
  generalization** of what SG-365 already does.

---

## 12. Suggested English wording (for the paper)

> The filtering problem is posed in local exponent space with an
> integrated-random-walk latent trend (level and velocity). The observation model
> is not a universal Gaussian: Bitcoin price noise is well described by a nearly
> invariant Laplace law, whereas hashrate noise is better described by a Student-t
> distribution with heavier tails and potentially time-varying scale. Both
> likelihoods are Gaussian scale mixtures, so inference reduces to a robust Kalman
> smoother (EM), not a Gaussian filter. The measurement model is thus calibrated
> from the empirical noise laws rather than assumed.

---

## 13. Name & layout

`ssmix` = **S**tate-**S**pace + scale-**MIX**ture (see header). Folder `SSmix/`,
script `ssmix_HP.py`, this doc `ssmix_model.md`.

## 14. Build ladder (incremental — each step visually checkable)

| step | what | check |
|---|---|---|
| 1 | Gaussian IRW Kalman smoother on `n_P` | `z_P(t)` ±band reproduces SG-365 |
| 2 | same on `n_HR` | `z_H(t)` ±band reproduces SG-365 |
| 3 | swap price obs → **Laplace** (scale-mixture EM, b_P≈102) | outliers down-weighted; z_P robust |
| 4 | swap hashrate obs → **Student-t** (ν const, s_H(t) per cycle) | z_H robust; outlier weights sit on spikes |
| 5 | report velocity v(t) + per-point weights | diagnostics |
| 6 | M2: **TVP γ(t)** (Adriano-analog, robust) | γ(t)±band = time-varying coupling exponent |
| 7 | M3: bivariate coupled SSM (common factor / cross-lag) | coupling — data-limited |
