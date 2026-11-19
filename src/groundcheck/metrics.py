"""Calibration and reliability metrics.

Plain-language summary: numbers that say how trustworthy a confidence score is.
ECE: on average, how far stated confidence is from actual accuracy (lower is
better). Brier: squared gap between confidence and right/wrong (lower is better).
AUROC: how well confidence separates right answers from wrong ones (0.5 = coin
flip, 1.0 = perfect). AURC: average error rate when the model is allowed to
abstain on its least-confident items first (lower is better).
"""
from __future__ import annotations

import numpy as np


def _clean(conf, correct):
    c = np.asarray(conf, float)
    y = np.asarray(correct, float)
    m = ~np.isnan(c)
    return np.clip(c[m], 0, 1), y[m]


def ece(conf, correct, n_bins: int = 15) -> float:
    c, y = _clean(conf, correct)
    if len(c) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(c, bins[1:-1], right=True), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(c[m].mean() - y[m].mean())
    return float(tot)


def brier(conf, correct) -> float:
    c, y = _clean(conf, correct)
    return float(np.mean((c - y) ** 2)) if len(c) else float("nan")


def auroc(conf, correct) -> float:
    c, y = _clean(conf, correct)
    if len(set(y.tolist())) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, c))


def aurc(conf, correct) -> float:
    c, y = _clean(conf, correct)
    if len(c) == 0:
        return float("nan")
    order = np.argsort(-c, kind="stable")
    err = 1 - y[order]
    risks = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risks.mean())


def reliability_bins(conf, correct, n_bins: int = 10):
    c, y = _clean(conf, correct)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(c, bins[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append(dict(lo=bins[b], hi=bins[b + 1], n=int(m.sum()),
                         conf=float(c[m].mean()) if m.any() else np.nan,
                         acc=float(y[m].mean()) if m.any() else np.nan))
    return rows


def summary(conf, correct) -> dict:
    c, y = _clean(conf, correct)
    return dict(n=int(len(c)), accuracy=float(y.mean()) if len(y) else float("nan"),
                mean_conf=float(c.mean()) if len(c) else float("nan"),
                ece=ece(conf, correct), brier=brier(conf, correct),
                auroc=auroc(conf, correct), aurc=aurc(conf, correct))


def bootstrap_ci(fn, conf, correct, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    c, y = _clean(conf, correct)
    if len(c) < 10:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(c), len(c))
        v = fn(c[i], y[i])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))
