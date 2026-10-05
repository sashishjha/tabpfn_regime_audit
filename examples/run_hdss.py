"""Run the regime audit on the Iganga-Mayuge HDSS verbal-autopsy labels (Zindi Climate Risk & Health data).
Usage: python examples/run_hdss.py path/to/Train.csv"""
import sys
import pandas as pd
from regime_audit import audit

df = pd.read_csv(sys.argv[1])
df["male"] = (df.gender == "Male").astype(int)
df["rural"] = (df.zone == "Rural").astype(int)
df["deathdate"] = pd.to_datetime(df.deathdate)
findings, p = audit(df, target="is_climate_sensitive", time_col="deathdate", feature_cols=["age", "male", "rural"], group_col="location")
print(f"{'subgroup':22s} {'break date':12s} {'n_before':>8s} {'n_after':>8s} {'rate_b':>7s} {'rate_a':>7s} {'z':>7s}")
for f in findings:
    print(f"{f.subgroup:22s} {str(f.date.date()):12s} {f.n_before:8d} {f.n_after:8d} {f.rate_before:7.2f} {f.rate_after:7.2f} {f.z:7.1f}")

# TabPFN is in-context: does choosing the context by regime help on the broken subgroup?
from regime_audit import context_ablation
top = next(f for f in findings if f.subgroup.startswith("0<=age"))
res = context_ablation(df, "is_climate_sensitive", "deathdate", ["age", "male", "rural"], top.date, eval_mask=(df.age < 3).to_numpy(), group_col="location")
print(f"\nContext ablation on children <3 y after {top.date.date()}:")
print(res.round(3).to_string(index=False))

# Did the population change at the same time?
from regime_audit import composition_shift
cs = composition_shift(df, "deathdate", top.date, "age", bins=[-1, 0, 4, 14, 49, 200], start="2009-01-01")
cs.index = ["age 0", "age 1-4", "age 5-14", "age 15-49", "age 50+"]
print(f"\nComposition shift around {top.date.date()} (deaths per year, 2009 onward):")
print(cs.round(2).to_string())
