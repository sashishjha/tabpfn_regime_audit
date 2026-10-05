"""Planted-break test with a cheap sklearn model (no TabPFN needed): the scan must recover the date and subgroup."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from regime_audit import audit


class Cheap:
    def __init__(self, seed): self.m = GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=seed)
    def fit(self, X, y): self.m.fit(X, y); return self
    def predict_proba(self, X): return self.m.predict_proba(X)


def make(n=3000, seed=0, break_date="2015-01-01"):
    rng = np.random.default_rng(seed)
    age = rng.integers(0, 80, n); sex = rng.integers(0, 2, n)
    t = pd.to_datetime("2008-01-01") + pd.to_timedelta(rng.integers(0, 365 * 10, n), unit="D")
    p = np.where(age < 5, 0.95, 0.4)                      # base rates
    p = np.where((age < 5) & (t >= pd.Timestamp(break_date)), 0.55, p)   # planted break for under-5s only
    return pd.DataFrame(dict(age=age, sex=sex, t=t, y=(rng.random(n) < p).astype(int)))


def test_recovers_planted_break():
    """Break planted for under-5s only (~180 of 3000 rows): subgroup, date (within 60 days) and direction must be recovered."""
    findings, _ = audit(make(), "y", "t", ["age", "sex"], model_factory=Cheap)
    top = findings[0]
    assert top.subgroup == "0<=age<5"
    assert abs((top.date - pd.Timestamp("2015-01-01")).days) < 60
    assert top.rate_before > 0.9 and top.rate_after < 0.6 and top.z < -5


def test_composition_shift_detects_population_change():
    from regime_audit import composition_shift
    rng = np.random.default_rng(1)
    t = pd.to_datetime("2008-01-01") + pd.to_timedelta(np.r_[rng.integers(0, 365 * 5, 1500), rng.integers(365 * 5, 365 * 8, 300)], unit="D")
    age = np.r_[rng.integers(0, 80, 1500), np.zeros(300, dtype=int)]  # after year 5: only infants are recorded
    df = pd.DataFrame(dict(t=t, age=age))
    cs = composition_shift(df, "t", "2013-01-01", "age", bins=[-1, 0, 79])
    assert cs.loc[cs.index[0], "ratio_after_over_before"] > 5 and cs.loc[cs.index[1], "ratio_after_over_before"] < 0.05
