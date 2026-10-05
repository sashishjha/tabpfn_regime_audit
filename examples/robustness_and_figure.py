"""Robustness over seeds + README figure for the HDSS example.
Usage: python examples/robustness_and_figure.py data/Train.csv   (needs TABPFN_TOKEN; ~15 min on a laptop CPU)"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from regime_audit import audit, context_ablation, composition_shift

df = pd.read_csv(sys.argv[1])
df["male"] = (df.gender == "Male").astype(int); df["rural"] = (df.zone == "Rural").astype(int)
df["deathdate"] = pd.to_datetime(df.deathdate)
FEATS = ["age", "male", "rural"]
rows, ctx_rows = [], []
for seed in range(5):
    findings, _ = audit(df, "is_climate_sensitive", "deathdate", FEATS, group_col="location", seed=seed)
    f = next(x for x in findings if x.subgroup == "0<=age<1")
    rows.append(dict(seed=seed, date=f.date, rate_before=f.rate_before, rate_after=f.rate_after, z=f.z))
    c = context_ablation(df, "is_climate_sensitive", "deathdate", FEATS, f.date, (df.age < 3).to_numpy(), group_col="location", seed=seed)
    c["seed"] = seed; ctx_rows.append(c)
    print(f"seed {seed}: break {f.date.date()} {f.rate_before:.2f}->{f.rate_after:.2f} z={f.z:.1f}", flush=True)
R = pd.DataFrame(rows); C = pd.concat(ctx_rows)
R.to_csv("results/robustness_break.csv", index=False); C.to_csv("results/robustness_context.csv", index=False)
d = pd.to_datetime(R.date)
print(f"\nbreak date over 5 seeds: {d.min().date()} .. {d.max().date()} (median {d.sort_values().iloc[2].date()}); z = {R.z.mean():.1f} +/- {R.z.std():.1f}")
summ = C.groupby("context")[["logloss", "brier", "mean_pred"]].agg(["mean", "std"]).round(3)
print(summ.to_string()); summ.to_csv("results/robustness_context_summary.csv")

# ---- figure
bd = d.sort_values().iloc[2]
inf = df[df.age < 1].sort_values("deathdate")
roll = inf.set_index("deathdate").is_climate_sensitive.rolling("180D", min_periods=15).mean()
cs = composition_shift(df, "deathdate", bd, "age", bins=[-1, 0, 4, 14, 49, 200], start="2009-01-01")
labels = ["0", "1-4", "5-14", "15-49", "50+"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(roll.index, roll.values, color="#1d4ed8"); ax[0].axvline(bd, color="#dc2626", ls="--", label=f"detected break {bd.date()}")
ax[0].set_title("Infants (<1 y): share labelled climate-sensitive\n(180-day rolling)"); ax[0].set_ylim(0, 1.05); ax[0].legend(loc="lower left")
x = np.arange(len(labels)); w = 0.38
ax[1].bar(x - w/2, cs.per_year_before.values, w, label="before", color="#64748b"); ax[1].bar(x + w/2, cs.per_year_after.values, w, label="after", color="#dc2626")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].set_title("Deaths per year by age group\n(who is in the data changed too)"); ax[1].set_xlabel("age (years)"); ax[1].legend()
cm = C.groupby("context")[["logloss"]].agg(["mean", "std"]); cm.columns = ["m", "s"]; cm = cm.loc[["all rows", "post-break only", "pre-break only"]]
ax[2].bar(cm.index, cm.m, yerr=cm.s, color=["#64748b", "#16a34a", "#dc2626"], capsize=4); ax[2].set_title("TabPFN-3.5 context choice\nheld-out log-loss, children <3 y after the break (5 seeds)")
ax[2].set_ylabel("log-loss (lower is better)")
plt.tight_layout(); plt.savefig("results/figure_regime.png", dpi=140); print("saved results/figure_regime.png")
