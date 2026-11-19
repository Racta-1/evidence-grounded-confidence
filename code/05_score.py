"""05_score.py: score every finished model run and write the stats file.

Plain-language summary: turns each model's raw answers into confidence scores,
applies the evidence-grounded adjustment, and measures how trustworthy each
confidence measure is. Writes, per run:
  results/<run>/item_scores.csv   one row per question (what you spot-check)
  results/<run>/metrics.json      summary numbers
and, across all runs, paper/stats.json (every number any document quotes) and
paper/tables/metrics_table.csv / .md.

Run:  python code/05_score.py                (all runs in results/)
      python code/05_score.py --agg max      (decision D6 alternative)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from groundcheck.score import item_frame, score  # noqa: E402
from groundcheck.suite import Suite, is_private  # noqa: E402


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="mean", choices=["mean", "max"])
    a = ap.parse_args()
    suite = Suite(ROOT)
    stats = {"suite": suite_stats(suite), "runs": {}}
    # summary counts and metrics on MIMIC are shareable; its rows and text are not
    rows = []
    for raw in sorted(glob.glob(os.path.join(ROOT, "results", "*", "raw.jsonl"))):
        run = os.path.basename(os.path.dirname(raw))
        meta = json.load(open(os.path.join(os.path.dirname(raw), "run_meta.json")))
        df = item_frame(raw, suite)
        if df.empty:
            continue
        df, res = score(df, agg=a.agg)
        priv = df.item_id.map(is_private)
        df[~priv].to_csv(os.path.join(os.path.dirname(raw), "item_scores.csv"), index=False)
        if priv.any():  # MIMIC rows stay on this machine (data use agreement)
            df[priv].to_csv(os.path.join(os.path.dirname(raw), "item_scores_private.csv"), index=False)
        res["model_id"] = meta["config"].get("model_id")
        res["n_items_scored"] = int(len(df))
        res["n_items_planned"] = meta["n_items"]
        res["complete"] = "finished" in meta
        json.dump(clean(res), open(os.path.join(os.path.dirname(raw), "metrics.json"), "w"), indent=2)
        stats["runs"][run] = clean(res)
        for ds, r in res["by_dataset"].items():
            for m, s in r["methods"].items():
                rows.append(dict(run=run, dataset=ds, method=m, n=s["n"], accuracy=s["accuracy"],
                                 mean_conf=s["mean_conf"], ece=s["ece"], brier=s["brier"],
                                 auroc=s["auroc"], aurc=s["aurc"]))
        print(f"scored {run}: {len(df)} items")
    os.makedirs(os.path.join(ROOT, "paper", "tables"), exist_ok=True)
    t = pd.DataFrame(rows)
    if len(t):
        t.round(3).to_csv(os.path.join(ROOT, "paper", "tables", "metrics_table.csv"), index=False)
        with open(os.path.join(ROOT, "paper", "tables", "metrics_table.md"), "w") as f:
            f.write(t.round(3).to_markdown(index=False) if hasattr(t, "to_markdown") else t.round(3).to_string(index=False))
    json.dump(clean(stats), open(os.path.join(ROOT, "paper", "stats.json"), "w"), indent=2)
    print("wrote paper/stats.json")


def suite_stats(s: Suite) -> dict:
    v = s.variants.merge(s.items[["item_id", "dataset"]], on="item_id")
    ok = v[v.status.str.startswith("ok")]
    return dict(
        n_items=int(len(s.items)),
        items_by_dataset={k: int(x) for k, x in s.items.dataset.value_counts().items()},
        variants_ok={f"{d}|{var}": int(n) for (d, var), n in ok.groupby(["dataset", "variant"]).size().items()},
        n_variant_renderings=int(len(ok)),
    )


if __name__ == "__main__":
    main()
