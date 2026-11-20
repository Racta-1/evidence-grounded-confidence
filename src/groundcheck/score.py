"""Turn raw model outputs into per-item scores and summary metrics.

Plain-language summary: reads what a model answered (raw.jsonl), works out each
confidence measure and the evidence-grounded adjustment, tunes the adjustment on
the calibration half, and measures everything on the test half.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import method as M
from . import prompts as PR
from . import metrics as MT
from .suite import EVIDENCE_REMOVING, Suite

VARS = ("original", "mask_region", "mask_control", "swap_image", "remove_text")


def item_frame(raw_path: str, suite: Suite) -> pd.DataFrame:
    import os
    recs = []
    for p in (raw_path, raw_path.replace("raw.jsonl", "raw_private.jsonl")):
        if os.path.exists(p):
            recs += [json.loads(l) for l in open(p)]
    by = {}
    for r in recs:
        by.setdefault(r["item_id"], {})[r["variant"]] = r
    rows = []
    for iid, d in by.items():
        if "original" not in d:
            continue
        o = d["original"]
        it = suite.item(iid)
        opts = list(o["probs"].keys()) if "probs" in o else []
        row = dict(item_id=iid, dataset=it.dataset, split=it.split, answer_type=it.answer_type,
                   content_type=it.content_type, gold=it.answer, pred=o["pred"],
                   # re-marked at scoring time from the saved answer, so a change to the matching
                   # rule never needs a new GPU run
                   correct=bool(PR.is_correct(o["pred"], it.answer, it.answer_type)),
                   chance=M.chance_level(it.answer_type, len(opts)),
                   conf_token=o["probs"][o["pred"]] if "probs" in o else o["p_a0"],
                   conf_verbal=o.get("verbal_conf") if o.get("verbal_conf") is not None else np.nan,
                   conf_consistency=o.get("consistency", np.nan),
                   probs_original=json.dumps(o.get("probs", {})))
        for v in VARS:
            r = d.get(v)
            row[f"p_a0__{v}"] = r["p_a0"] if r else np.nan
            row[f"changed__{v}"] = (r["pred"] != o["pred"]) if (r and v != "original") else np.nan
            row[f"probs__{v}"] = json.dumps(r.get("probs", {})) if r else ""
        rows.append(row)
    df = pd.DataFrame(rows)
    surv = df.apply(lambda r: M.survival_per_variant(r), axis=1)
    for v in EVIDENCE_REMOVING:
        df[f"u__{v}"] = surv.map(lambda d: d.get(v, np.nan))
    df["u_mean"] = M.survival(df, agg="mean")
    df["u_max"] = M.survival(df, agg="max")
    return df


def _safe_fit(model, X, y):
    if len(set(np.asarray(y).tolist())) < 2 or len(y) < 10:
        return None
    try:
        return model.fit(X, y)
    except Exception:
        return None


def score(df: pd.DataFrame, agg: str = M.DEFAULT_AGG, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["u"] = df["u_mean"] if agg == "mean" else df["u_max"]
    cal, test = df[df.split == "cal"], df[df.split == "test"]
    fitted = {}
    # per-dataset fitting: models and datasets differ too much to share one lambda
    df["conf_platt"] = np.nan
    df["conf_groundcheck"] = np.nan
    df["conf_groundcheck_logistic"] = np.nan
    for ds in df.dataset.unique():
        c = cal[cal.dataset == ds]
        idx = df.dataset == ds
        lam = M.fit_lambda(c.conf_token, c.u, c.correct) if len(c) else 0.0
        df.loc[idx, "conf_groundcheck"] = M.groundcheck_adjust(df.loc[idx, "conf_token"], df.loc[idx, "u"], lam)
        pl = _safe_fit(M.Platt(), c.conf_token.values, c.correct.astype(int).values)
        if pl is not None:
            df.loc[idx, "conf_platt"] = pl.predict(df.loc[idx, "conf_token"].values)
            # GroundCheck on top of Platt: recalibrate, then apply the evidence adjustment
        lg = _safe_fit(M.GroundCheckLogistic(), c, c.correct.astype(int).values)
        if lg is not None:
            df.loc[idx, "conf_groundcheck_logistic"] = lg.predict(df[idx])
        fitted[ds] = dict(lambda_=lam, n_cal=int(len(c)), platt=pl is not None, logistic=lg is not None)
    df["flag"] = M.flag(df.conf_token, df.u)
    test = df[df.split == "test"]
    methods = {"Token probability": "conf_token", "Verbalized": "conf_verbal",
               "Sampling consistency": "conf_consistency", "Platt-scaled token prob.": "conf_platt",
               "GroundCheck (evidence-adjusted)": "conf_groundcheck", "GroundCheck-logistic": "conf_groundcheck_logistic"}
    out = {"fitted": fitted, "aggregation": agg, "by_dataset": {}}
    for ds in sorted(test.dataset.unique()):
        t = test[test.dataset == ds]
        res = {"n_test": int(len(t)), "accuracy": float(t.correct.mean()), "methods": {}}
        for name, col in methods.items():
            s = MT.summary(t[col], t.correct)
            s["ece_ci"] = MT.bootstrap_ci(MT.ece, t[col], t.correct, n=500, seed=seed)
            s["auroc_ci"] = MT.bootstrap_ci(MT.auroc, t[col], t.correct, n=500, seed=seed)
            res["methods"][name] = s
        # evidence sensitivity
        sens = {}
        for v in ("mask_region", "mask_control", "swap_image", "remove_text"):
            m = t[f"p_a0__{v}"].notna()
            if m.any():
                sens[v] = dict(n=int(m.sum()),
                               mean_conf_original=float(t.loc[m, "p_a0__original"].mean()),
                               mean_conf_variant=float(t.loc[m, f"p_a0__{v}"].mean()),
                               answer_changed=float(t.loc[m, f"changed__{v}"].astype(float).mean()),
                               mean_survival=float(t.loc[m, f"u__{v}"].mean()) if f"u__{v}" in t else float("nan"))
        res["evidence_sensitivity"] = sens
        f = t.flag
        res["flags"] = dict(n_flagged=int(f.sum()), rate=float(f.mean()),
                            acc_flagged=float(t[f].correct.mean()) if f.any() else float("nan"),
                            acc_unflagged=float(t[~f].correct.mean()) if (~f).any() else float("nan"))
        res["conformal"] = conformal_block(df[df.dataset == ds])
        out["by_dataset"][ds] = res
    return df, out


def conformal_block(d: pd.DataFrame, alpha: float = 0.1) -> dict:
    d = d[d.answer_type.isin(["closed", "mcq"])]
    if d.empty:
        return {}
    cal, test = d[d.split == "cal"], d[d.split == "test"]
    pt = [json.loads(p).get(g if at == "mcq" else g.lower(), 0.0)
          for p, g, at in zip(cal.probs_original, cal.gold, cal.answer_type)]
    q = M.conformal_qhat(np.array(pt), alpha)
    if np.isnan(q) or test.empty:
        return {}
    res = {"alpha": alpha, "qhat": q}
    for v in ("original", "mask_region", "mask_control", "swap_image", "remove_text"):
        col = f"probs__{v}"
        tt = test[test[col] != ""]
        if tt.empty:
            continue
        sets = M.conformal_sets([json.loads(p) for p in tt[col]], q)
        gold = [g if at == "mcq" else g.lower() for g, at in zip(tt.gold, tt.answer_type)]
        res[v] = dict(n=int(len(tt)), mean_set_size=float(np.mean([len(s) for s in sets])),
                      coverage=float(np.mean([g in s for g, s in zip(gold, sets)])))
    return res
