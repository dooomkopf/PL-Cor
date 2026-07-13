# PL-Cor — Bitcoin Power-Law Correlation
# Collab-work w/ A. Pecere (@ZeitgeistExplo1 on X), reg. Kaman RTS in price space

Tools for studying Bitcoin's **power-law scaling laws** and the coupling between
them — in particular the price exponent `n_P(t)` and the hashrate exponent
`n_HR(t)`. It treats Bitcoin as a coupled dynamical system.

Each metric `X` (price, hashrate, …) grows roughly as a power law of network
age `t` (days since the genesis block):

```
X(t) ≈ A · t^n
```

The **local exponent**, computed between consecutive days (natural log), is the
central quantity:

```
n(t) = log( X(t+1) / X(t) ) / log( t(t+1) / t(t) )
```

Working in this *exponent space* (rather than the price space) matters
because regressing two trending level series directly can give spurious results;
the residual stationarity test in `stationarity.py` checks whether a given level
relation is genuine or spurious.

## Math explainer draft

### `impulse-response-DRAFT.pdf`
Impulse-response (convolution) model between the hashrate and price exponents
`n_HR(t)` / `n_P(t)`: a Tikhonov-smoothed lag-kernel estimator with a
closed-form solution, cross-validation in time blocks, and
block-wild-bootstrap bands. Main result: the price->hashrate response is
permanent (step response `H(200) ~ +0.27..+0.33`, probe-invariant), whereas
hashrate->price is transient with a negative tail. Includes robustness checks
(probe grid, bear-regime interaction, L2 spring control, phase analysis) and
the mathematical background. (Draft, English.)

## Quick start

```
git clone https://github.com/dooomkopf/PL-Cor.git
cd PL-Cor
pip install -r requirements.txt   # Python dependencies only
python fetch_coinmetrics.py       # download cm_data.csv (CoinMetrics Community API)
python stationarity.py            # run any analysis (see the table below)
```

`pip install -r requirements.txt` installs only the Python packages — you still
clone the files and fetch the data (one `fetch_coinmetrics.py` run) yourself.

## Repository layout (convention)

Flat layout. One self-contained script per analysis:

```
name.py            # an executable analysis (run it directly)
name_README.txt    # what it does, the math, usage and options
```

Shared modules (imported by several analyses):

| module | role |
|--------|------|
| `data_io.py`  | load the CoinMetrics CSV, network age, positivity mask (no math) |
| `dailynHR.py` | `daily_exponent()` — the core local-exponent formula; cycle/colour constants |
| `scaling.py`  | `loglog_ols`, `loglog_quantile` — power-law fits in log-log space |

Every script puts its **parameters in a block at the top**, and every relevant
piece of **mathematics is commented**.

## Install

```
pip install -r requirements.txt
```

## Get the data

The analyses read `cm_data.csv` (`date, P, N, H` = price, active addresses,
hashrate) from the CoinMetrics **Community** API. It is not committed — fetch it
once:

```
python fetch_coinmetrics.py
```

## Analyses

Each script has a `*_README.txt` with the full idea, math and options. Run any
analysis with `--help`.

### `stationarity.py`
ADF/KPSS unit-root tests on the price/hashrate levels and the power-law-fit
residuals (reports the integration order). → [details](stationarity_README.txt)

![stationarity.py output](stationarity.png)

### `dailyn_P_evol.py`
Price exponent `n_P(t)`: raw daily points + SG-91/181/365 smooths. → [details](dailyn_P_evol_README.txt)

![dailyn_P_evol.py output](dailyn_P_evol.png)

### `dailyn_H_evol.py`
Hashrate exponent `n_HR(t)`: raw daily points + SG-91/181/365 smooths. → [details](dailyn_H_evol_README.txt)

![dailyn_H_evol.py output](dailyn_H_evol.png)

### `dailyn_HP_evol.py`
The SG-smoothed `n_P` and `n_HR` overlaid in one panel. → [details](dailyn_HP_evol_README.txt)

![dailyn_HP_evol.py output](dailyn_HP_evol.png)

### `scale_inv_HP.py`
Scale-invariance test: log-ratio scatter of random time pairs, price vs hashrate. → [details](scale_inv_HP_README.txt)

![scale_inv_HP.py output](scale_inv_HP.png)

### `dailynHR_distfit.py`
The noise measurement behind ssmix: MLE fits of Normal / Laplace / Student-t /
Generalized Normal / Cauchy to the daily `n_HR`, per halving cycle, compared
via AIC/BIC (overall), KS (center) and Anderson-Darling (tails). The winning
per-cycle Student-t parameters are the **fixed** noise channel `HASH_CYC` in
`ssmix_HP.py`: the fitted location is discarded downstream, while the measured
width and shape are used drift-free and never re-fitted in-model. The global
pooled fit is shown separately because it hides the cycle-dependent widths.
→ [details](dailynHR_distfit_README.txt)

![dailynHR_distfit.py per-cycle output](dailynHR_distfit_cycles.png)

![dailynHR_distfit.py global output](dailynHR_distfit_global.png)

### `noise_movav_H.py`
Validation of that noise law in the H diagnostic: Monte-Carlo paths whose daily
steps contain only the measured exponent noise (`ln H_t = ln H_{t-1} + n ·
ln(t/(t−1))`, `n` ~ per-cycle Student-t, drift-free), around an invisible
centered 30-day moving-average anchor of the real hashrate. A good noise law
⇒ the Q1..Q99 band envelopes the real scatter. The same MC construction
carries the measured exponent noise into the H/P diagnostic for the ssmix
coupling estimator; `log` appears only as the arithmetic power-law transform
`d log(X) = n_X d log(t)`. → [details](noise_movav_H_README.txt)

![noise_movav_H.py output](noise_movav_H.png)

### `ssmix_TVP.py`
Time-varying H/P coupling `γ(t)`. Panel (a) is the primary test in the
`n_H/n_P`-Raum: the per-cycle, noise-aware TVP Kalman estimate `γ_exp(t)` and
its free-intercept check. Panel (b) is the H/P diagnostic: Adrianos frozen
weekly Gaussian RTS reconstruction, `mod RTS` with the measured Monte-Carlo
noise variance, and `integrated γ_exp` as the back-transform of the primary
estimate. The Student-t shape is already represented by the MC draws, so
`mod RTS` applies no additional residual-dependent Student-t reweighting.
→ [details](ssmix_TVP_README.txt)

![ssmix_TVP.py output](ssmix_TVP.png)

## Data source / attribution

Daily metrics from the [CoinMetrics Community API](https://coinmetrics.io/community-network-data/).
