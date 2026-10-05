# regime_audit — find label-regime breaks in tabular time series with TabPFN-3.5

**Problem.** In health surveillance, fraud, credit and many other tabular settings the *labelling rule* can change silently
(a new instrument, a new coding algorithm, a policy change). Models trained across the break learn an average of two regimes
and are mis-calibrated on the new one — and nobody notices because aggregate metrics look fine.

**What this does.**
1. Fits a **time-free** TabPFN-3.5 model `P(y | x)` out of fold (group-aware folds supported). Dates are *not* given to the model.
2. If the labelling rule changed for some subgroup, that subgroup's residuals `y − p̂` jump at that date. The tool scans every
   candidate date in every automatically discovered subgroup (all contiguous ranges of numeric features, every category) and
   ranks by a residual-shift z-score, then refines the date.
3. **TabPFN-specific test:** TabPFN is in-context, so *choosing the context rows by regime* is a free intervention. `context_ablation`
   compares context = all rows / post-break only / pre-break only on held-out groups.

## Result on real data (Iganga-Mayuge HDSS verbal-autopsy labels, 3,146 deaths 2007-2022)
Features: age, sex, zone only. Location-grouped 5-fold OOF predictions. See `results/hdss_audit.txt`.

| subgroup | break date | rate before → after | z |
|---|---|---|---|
| age < 1 year | **2017-06-06** | **0.97 → 0.56** | −17.5 |
| age < 3 | 2017-06-06 | 0.96 → 0.59 | −14.8 |
| all deaths | 2017-05-28 | 0.69 → 0.53 | −12.2 |

The blind scan recovers an abrupt change in how infant deaths are classified as "climate-sensitive" in mid-2017
(consistent with a change of the verbal-autopsy cause-assignment instrument). Context choice matters, on held-out villages
(children < 3 y after the break, n = 536):

| TabPFN context | log-loss | Brier | mean prediction (true rate 0.586) |
|---|---|---|---|
| all rows | 0.785 | 0.284 | 0.800 |
| **post-break rows only** | **0.675** | **0.242** | **0.594** |
| pre-break rows only | 1.446 | 0.389 | 0.966 |

Choosing the context by regime cuts log-loss by 14% and restores calibration, with no retraining. Ranking (AUC ≈ 0.5 inside the
post-break group) does not improve: the post-break labels are not predictable from age/sex/zone at all.

## Quick start
```bash
git clone https://github.com/sashishjha/tabpfn_regime_audit && cd tabpfn_regime_audit
uv sync                                   # Python >= 3.12 (installs the package in editable mode)
export TABPFN_TOKEN=...                   # one-time TabPFN licence acceptance, see https://ux.priorlabs.ai
PYTHONPATH=src uv run python examples/run_synthetic.py        # no data needed: planted break is recovered
PYTHONPATH=src uv run python examples/run_hdss.py data/Train.csv
PYTHONPATH=src uv run --with pytest python -m pytest -q tests  # cheap sklearn model, no TabPFN needed
```
```python
from regime_audit import audit, context_ablation
findings, p_oof = audit(df, target="y", time_col="date", feature_cols=["age", "sex"], group_col="site")
```
On CPU with more than 1,000 rows set `TABPFN_ALLOW_CPU_LARGE_DATASET=1` (done automatically by the package); a GPU is faster.

## Caveats
* Many subgroups × dates are scanned, so z-scores are optimistic; treat them as a ranking and confirm top findings with a hold-out
  or permutation test. The planted-break test shows the scan recovers a 180-row subgroup within 11 days.
* The scan finds *that* a rule changed, not *why*; domain knowledge is needed to interpret it.
* Feature columns must exclude time features, otherwise the model absorbs the break.

## Data and licence
Code: Apache-2.0 (`LICENSE`). `data/` holds the HDSS training table released with the Zindi "Climate Risk and Health Prediction
Challenge" under CC-BY-SA 4.0; see `data/README.md`.
