"""Residual changepoint scan on top of out-of-fold TabPFN-3.5 predictions.

Idea: fit a time-FREE model P(y | x) (e.g. age, sex, zone) with TabPFN-3.5 out of fold. If the labelling rule or the
classification instrument changes at some date for some subgroup, the residuals y - p_hat of that subgroup jump at that date.
We scan every candidate date in every automatically discovered subgroup and rank by a z-score of the residual shift.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")


@dataclass
class BreakFinding:
    subgroup: str
    date: pd.Timestamp
    n_before: int
    n_after: int
    rate_before: float
    rate_after: float
    resid_before: float
    resid_after: float
    z: float


def make_tabpfn(random_state: int = 0, device: str = "cpu"):
    """TabPFN-3.5 (the default model version of tabpfn>=9)."""
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    return TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, random_state=random_state, device=device,
                                                       n_estimators=4)


def oof_probabilities(X: pd.DataFrame, y: np.ndarray, groups=None, n_splits: int = 5, seed: int = 0, model_factory=make_tabpfn):
    """Out-of-fold probabilities; group-aware if groups are given."""
    oof = np.zeros(len(X))
    splitter = (StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups) if groups is not None
                else StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y))
    for a, b in splitter:
        m = model_factory(seed)
        m.fit(X.iloc[a].to_numpy(dtype=float), y[a])
        oof[b] = m.predict_proba(X.iloc[b].to_numpy(dtype=float))[:, 1]
    return oof


def candidate_subgroups(df: pd.DataFrame, feature_cols, fine_bins: int = 16):
    """Subgroups from single features: categorical/binary -> each value; numeric -> EVERY contiguous range of `fine_bins`
    quantile bins (so a break confined to a narrow slice, e.g. infants, is not diluted); plus the full sample."""
    yield "all", np.ones(len(df), dtype=bool)
    for c in feature_cols:
        v = df[c]
        if v.nunique() <= 6:
            for val in sorted(v.unique()):
                yield f"{c}=={val}", (v == val).to_numpy()
            continue
        edges = np.unique(np.quantile(v, np.linspace(0, 1, fine_bins + 1)))
        edges[-1] = np.nextafter(edges[-1], np.inf)
        for i in range(len(edges) - 1):
            for j in range(i + 1, len(edges)):
                if i == 0 and j == len(edges) - 1:
                    continue  # the full range is "all"
                yield f"{edges[i]:g}<={c}<{edges[j]:g}", ((v >= edges[i]) & (v < edges[j])).to_numpy()


def scan_breaks(df, y, p, time_col, subgroup_masks, min_side: int = 40, grid: int = 60):
    """For each subgroup scan candidate dates; z = shift of mean residual (y - p) after vs before the date."""
    t = pd.to_datetime(df[time_col]).to_numpy()
    out = []
    resid = y - p
    for name, m in subgroup_masks:
        idx = np.where(m)[0]
        if len(idx) < 2 * min_side:
            continue
        order = idx[np.argsort(t[idx])]
        r, yy, tt = resid[order], y[order], t[order]
        n = len(r)
        cands = np.unique(np.linspace(min_side, n - min_side, grid).astype(int))
        best = None
        for k in cands:
            rb, ra = r[:k], r[k:]
            se = np.sqrt(rb.var() / len(rb) + ra.var() / len(ra) + 1e-12)
            z = (ra.mean() - rb.mean()) / se
            if best is None or abs(z) > abs(best[0]):
                best = (z, k)
        # refine: scan every position between the neighbouring grid points of the coarse optimum
        step = max(1, int(cands[1] - cands[0])) if len(cands) > 1 else 1
        for k in range(max(min_side, best[1] - step), min(n - min_side, best[1] + step) + 1):
            rb, ra = r[:k], r[k:]
            z = (ra.mean() - rb.mean()) / np.sqrt(rb.var() / len(rb) + ra.var() / len(ra) + 1e-12)
            if abs(z) > abs(best[0]):
                best = (z, k)
        z, k = best
        out.append(BreakFinding(name, pd.Timestamp(tt[k]), k, n - k, yy[:k].mean(), yy[k:].mean(), r[:k].mean(), r[k:].mean(), z))
    return sorted(out, key=lambda f: -abs(f.z))


def audit(df: pd.DataFrame, target: str, time_col: str, feature_cols, group_col: str | None = None, seed: int = 0,
          model_factory=make_tabpfn, top: int = 10):
    """Returns (findings, p_oof). feature_cols must NOT contain time features, so a regime change shows up as a residual shift."""
    y = df[target].to_numpy(dtype=int)
    groups = df[group_col].to_numpy() if group_col else None
    p = oof_probabilities(df[list(feature_cols)], y, groups, seed=seed, model_factory=model_factory)
    findings = scan_breaks(df, y, p, time_col, list(candidate_subgroups(df, feature_cols)))
    return findings[:top], p


def context_ablation(df, target, time_col, feature_cols, break_date, eval_mask, group_col=None, seed=0, n_splits=5,
                     model_factory=make_tabpfn):
    """TabPFN is in-context: does CHOOSING the context by regime help? Evaluate on held-out rows that are in eval_mask and after
    break_date, using context = all rows / only post-break rows / only pre-break rows. Returns log-loss, Brier, AUC per strategy."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    y = df[target].to_numpy(dtype=int)
    X = df[list(feature_cols)].to_numpy(dtype=float)
    after = (pd.to_datetime(df[time_col]) >= pd.Timestamp(break_date)).to_numpy()
    target_rows = np.asarray(eval_mask) & after
    splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, df[group_col].to_numpy() if group_col else None)
    preds = {k: np.full(len(df), np.nan) for k in ("all rows", "post-break only", "pre-break only")}
    for a, b in splitter:
        b = b[target_rows[b]]
        if len(b) == 0:
            continue
        for k, ctx in (("all rows", a), ("post-break only", a[after[a]]), ("pre-break only", a[~after[a]])):
            m = model_factory(seed)
            m.fit(X[ctx], y[ctx])
            preds[k][b] = m.predict_proba(X[b])[:, 1]
    rows = []
    for k, p in preds.items():
        m = ~np.isnan(p)
        rows.append(dict(context=k, n_eval=int(m.sum()), logloss=log_loss(y[m], np.clip(p[m], 1e-4, 1 - 1e-4)),
                         brier=brier_score_loss(y[m], p[m]), auc=roc_auc_score(y[m], p[m]) if len(set(y[m])) > 1 else float("nan"),
                         mean_pred=float(p[m].mean()), mean_label=float(y[m].mean())))
    return pd.DataFrame(rows)


def composition_shift(df: pd.DataFrame, time_col: str, break_date, group_col: str, bins=None, start=None):
    """Did the *population* change at the break as well as the labels? Annualised counts per group before vs after the break.
    A label-rate break that coincides with a collapse/surge in group sizes points to a change of data source, not only of coding."""
    t = pd.to_datetime(df[time_col])
    b = pd.Timestamp(break_date)
    start = pd.Timestamp(start) if start is not None else t.min()
    years_pre = max((b - start).days / 365.25, 1e-9)
    years_post = max((t.max() - b).days / 365.25, 1e-9)
    g = pd.cut(df[group_col], bins) if bins is not None else df[group_col]
    pre = g[(t < b) & (t >= start)].value_counts().sort_index()
    post = g[t >= b].value_counts().reindex(pre.index).fillna(0)
    out = pd.DataFrame({"per_year_before": pre / years_pre, "per_year_after": post / years_post})
    out["ratio_after_over_before"] = out.per_year_after / out.per_year_before
    out["share_before"] = pre / pre.sum()
    out["share_after"] = post / post.sum()
    return out
