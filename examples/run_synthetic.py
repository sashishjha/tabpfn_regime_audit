"""Self-contained demo (no data download): plant a label-regime break and recover it with TabPFN-3.5.
Usage: python examples/run_synthetic.py"""
import numpy as np
import pandas as pd
from regime_audit import audit

rng = np.random.default_rng(0)
n = 3000
age = rng.integers(0, 80, n); sex = rng.integers(0, 2, n)
t = pd.to_datetime("2008-01-01") + pd.to_timedelta(rng.integers(0, 365 * 10, n), unit="D")
p = np.where(age < 5, 0.95, 0.40)
p = np.where((age < 5) & (t >= pd.Timestamp("2015-01-01")), 0.55, p)  # the planted break: under-5s only, from 2015-01-01
df = pd.DataFrame(dict(age=age, sex=sex, t=t, y=(rng.random(n) < p).astype(int)))
findings, _ = audit(df, "y", "t", ["age", "sex"])
for f in findings[:5]:
    print(f"{f.subgroup:16s} {f.date.date()}  rate {f.rate_before:.2f} -> {f.rate_after:.2f}  z={f.z:.1f}")
