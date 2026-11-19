"""The evidence-grounded confidence method (working name EGC) and the baselines.

Plain-language summary:
  1. Ask the model a question and note how confident it is in its answer.
  2. Take the evidence away (hide the region, swap the image, remove the story)
     and measure how much of that confidence in the SAME answer survives.
     Confidence that survives with the evidence gone was never resting on it.
  3. Flag answers that are highly confident AND keep that confidence without
     their evidence.
  4. Lower the reported confidence in proportion to how much survived.
     How strongly to lower it is learned on the calibration half of the data
     and then tested on the other half, so the method is never graded on the
     data it was tuned on.

Baselines compared against: token probability, verbalized (stated) confidence,
sampling consistency, conformal prediction sets, and Platt scaling of token
probability (the fair "just recalibrate it" comparison).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .suite import EVIDENCE_REMOVING

EPS = 1e-6

# Author decisions (docs/VERIFY_CHECKLIST.md): D6 aggregation, D7 flag thresholds
DEFAULT_AGG = "mean"
TAU_CONF = 0.8
TAU_SURVIVE = 0.8


def chance_level(answer_type: str, n_options: int) -> float:
    return 1.0 / n_options if answer_type in ("closed", "mcq") and n_options > 0 else 0.0


def survival_per_variant(row: pd.Series, use_control: bool = True) -> dict[str, float]:
    """Share of confidence in the original answer that survives each evidence removal,
    with chance level subtracted (0 = fell to chance or below, 1 = fully survived)."""
    chance = row.get("chance", 0.0)
    out = {}
    for v in EVIDENCE_REMOVING:
        s = row.get(f"p_a0__{v}", np.nan)
        if s is None or np.isnan(s):
            continue
        ref = row.get("p_a0__original", np.nan)
        if v == "mask_region" and use_control:
            c = row.get("p_a0__mask_control", np.nan)
            if c is not None and not np.isnan(c):
                ref = c
        denom = ref - chance
        if np.isnan(ref) or denom <= EPS:
            out[v] = 1.0 if s >= ref else 0.0  # confidence already at chance: nothing to lose
            continue
        out[v] = float(np.clip((s - chance) / denom, 0.0, 1.0))
    return out


def survival(df: pd.DataFrame, agg: str = DEFAULT_AGG, use_control: bool = True) -> pd.Series:
    def one(r):
        d = survival_per_variant(r, use_control)
        if not d:
            return np.nan
        vals = list(d.values())
        return float(np.max(vals) if agg == "max" else np.mean(vals))
    return df.apply(one, axis=1)


def egc_simple(c0, u, lam: float):
    c0 = np.asarray(c0, float)
    u = np.nan_to_num(np.asarray(u, float), nan=0.0)
    return np.clip(c0 * (1 - lam * u), 0, 1)


def fit_lambda(c0, u, correct, grid=np.linspace(0, 1, 21)) -> float:
    y = np.asarray(correct, float)
    best, best_l = np.inf, 0.0
    for lam in grid:
        b = np.nanmean((egc_simple(c0, u, lam) - y) ** 2)
        if b < best - 1e-12:
            best, best_l = b, float(lam)
    return best_l


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Platt:
    """Logistic recalibration of one confidence score (baseline)."""

    def fit(self, c, y):
        from sklearn.linear_model import LogisticRegression
        c = np.asarray(c, float)
        m = ~np.isnan(c)
        self.m = LogisticRegression(C=1e4).fit(_logit(c[m]).reshape(-1, 1), np.asarray(y)[m])
        return self

    def predict(self, c):
        c = np.asarray(c, float)
        out = np.full(len(c), np.nan)
        m = ~np.isnan(c)
        if m.any():
            out[m] = self.m.predict_proba(_logit(c[m]).reshape(-1, 1))[:, 1]
        return out


class EGCLogistic:
    """Learned variant: predicts correctness from the original confidence plus
    how much confidence survived each evidence removal."""

    def _X(self, df):
        cols = [_logit(df["conf_token"].values)]
        for v in EVIDENCE_REMOVING:
            s = df.get(f"u__{v}")
            s = np.full(len(df), np.nan) if s is None else s.values.astype(float)
            cols.append(np.nan_to_num(s, nan=0.0))
            cols.append(np.isnan(s).astype(float))
        return np.column_stack(cols)

    def fit(self, df, y):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=1.0, max_iter=1000).fit(self._X(df), np.asarray(y))
        return self

    def predict(self, df):
        return self.m.predict_proba(self._X(df))[:, 1]


def flag(c0, u, tau_c: float = TAU_CONF, tau_u: float = TAU_SURVIVE):
    c0 = np.asarray(c0, float)
    u = np.asarray(u, float)
    return (c0 >= tau_c) & (np.nan_to_num(u, nan=0.0) >= tau_u)


# ---------------------------------------------------------------------------
# Conformal prediction (closed-set items): split conformal, LAC score
# ---------------------------------------------------------------------------
def conformal_qhat(probs_true: np.ndarray, alpha: float = 0.1) -> float:
    s = 1 - np.asarray(probs_true, float)
    n = len(s)
    if n == 0:
        return np.nan
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(s)[min(k, n) - 1])


def conformal_sets(prob_dicts: list[dict], qhat: float) -> list[list[str]]:
    return [[o for o, p in d.items() if p >= 1 - qhat] for d in prob_dicts]
