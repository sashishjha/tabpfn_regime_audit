"""Bootstrap uncertainty of the infant break date (resample infant rows, rescan the best split). No TabPFN needed.
Usage: python examples/bootstrap_break_date.py data/Train.csv"""
import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1]); df["d"] = pd.to_datetime(df.deathdate)
g = df[df.age < 1].sort_values("d").reset_index(drop=True)
rng = np.random.default_rng(0)
def best_split(y, t, min_side=40):
    r = y - y.mean(); n = len(y); best = (0, None)
    for k in range(min_side, n - min_side):
        rb, ra = r[:k], r[k:]
        z = (ra.mean() - rb.mean()) / np.sqrt(rb.var() / k + ra.var() / (n - k) + 1e-12)
        if abs(z) > abs(best[0]): best = (z, k)
    return best[0], t[best[1]]
y = g.is_climate_sensitive.to_numpy(float); t = g.d.to_numpy()
z0, d0 = best_split(y, t); print(f"point estimate: {pd.Timestamp(d0).date()} z={z0:.1f} (n={len(g)})")
ds = []
for _ in range(500):
    i = np.sort(rng.integers(0, len(g), len(g)))
    ds.append(pd.Timestamp(best_split(y[i], t[i])[1]))
ds = pd.Series(ds)
print("bootstrap 5%/50%/95% of break date:", ds.quantile(0.05).date(), ds.quantile(0.5).date(), ds.quantile(0.95).date())
