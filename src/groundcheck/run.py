"""Run one model over the suite and save every raw output.

Plain-language summary: for each question, show the model the original and each
altered version, and save what it answered and how confident it was. Saving is
line by line, so an interrupted run picks up where it stopped.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import prompts as PR
from .perturb import seed_for
from .suite import Suite, is_private


def select_items(suite: Suite, datasets, splits, limit_per_dataset=None, require_variant=None, seed="pilot"):
    it = suite.items
    it = it[it.dataset.isin(datasets) & it.split.isin(splits)].copy()
    if require_variant:
        ok = suite.variants[(suite.variants.variant.isin(require_variant)) & (suite.variants.status.str.startswith("ok"))]
        keep = set(ok.item_id)
        it = it[(~it.dataset.isin(require_variant_datasets(suite, require_variant))) | it.item_id.isin(keep)]
    it["_order"] = it.item_id.map(lambda i: seed_for(i, seed))
    it = it.sort_values("_order")
    if limit_per_dataset:
        it = it.groupby("dataset", group_keys=False).head(limit_per_dataset)
    return it.item_id.tolist()


def require_variant_datasets(suite, variants):
    v = suite.variants.merge(suite.items[["item_id", "dataset"]], on="item_id")
    v = v[v.variant.isin(variants) & v.status.str.startswith("ok")]
    return set(v.dataset)


def private_path(path):
    return path.replace("raw.jsonl", "raw_private.jsonl")


def _done(path):
    done = set()
    for p in (path, private_path(path)):
        if not os.path.exists(p):
            continue
        for line in open(p):
            try:
                r = json.loads(line)
                done.add((r["item_id"], r["variant"]))
            except Exception:
                pass
    return done


def run_items(adapter, suite: Suite, item_ids, out_path, cfg: dict, log=print):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = _done(out_path)
    k = int(cfg.get("consistency_k", 5))
    temp = float(cfg.get("temperature", 1.0))
    verbal = bool(cfg.get("verbalized", True))
    max_new = int(cfg.get("max_new_tokens", 24))
    t0 = time.time()
    with open(out_path, "a") as f_pub, open(private_path(out_path), "a") as f_priv:
        for n_i, iid in enumerate(item_ids):
            f = f_priv if is_private(iid) else f_pub
            it = suite.item(iid)
            at = it.answer_type
            opts = PR.option_strings(at, json.loads(it.options) if it.options else [])
            a0 = None
            if hasattr(adapter, "begin_item"):
                adapter.begin_item(iid)
            for var in suite.available(iid):
                if var == "original" and (iid, var) in done:
                    # recover a0 from the saved original
                    for line in open(private_path(out_path) if is_private(iid) else out_path):
                        r = json.loads(line)
                        if r["item_id"] == iid and r["variant"] == "original":
                            a0 = r["a0"]
                            break
                    continue
                if (iid, var) in done:
                    continue
                rd = suite.render(iid, var)
                prompt = PR.build_prompt(rd.question, at)
                rec = dict(item_id=iid, variant=var, model=adapter.name, answer_type=at)
                if at in ("closed", "mcq"):
                    probs = adapter.option_probs(rd.image, prompt, opts)
                    pred = max(probs, key=probs.get)
                    rec.update(probs=probs, pred=pred)
                    if var == "original":
                        a0 = pred
                    rec["p_a0"] = probs.get(a0, float("nan"))
                else:
                    text, mlp = adapter.generate(rd.image, prompt, n=1, temperature=0.0, max_new_tokens=max_new)[0]
                    rec.update(gen_text=text, gen_mean_lp=mlp, pred=PR.parse_choice(text, at, opts))
                    if var == "original":
                        a0 = text if text else " "
                    lp, ntok = adapter.answer_logprob(rd.image, prompt, a0)
                    rec["p_a0"] = math.exp(lp / max(1, ntok))
                if var == "original":
                    rec["a0"] = a0
                    rec["correct"] = PR.is_correct(rec["pred"], it.answer, at)
                    if verbal:
                        vt, _ = adapter.generate(rd.image, PR.build_prompt(rd.question, at, verbalized=True),
                                                 n=1, temperature=0.0, max_new_tokens=max_new + 16)[0]
                        rec.update(verbal_text=vt, verbal_conf=PR.parse_verbal_conf(vt),
                                   verbal_pred=PR.parse_choice(vt, at, opts))
                    if k > 0:
                        samples = adapter.generate(rd.image, prompt, n=k, temperature=temp, max_new_tokens=max_new)
                        sp = [PR.parse_choice(s, at, opts) for s, _ in samples]
                        rec.update(samples=[s for s, _ in samples],
                                   consistency=float(np.mean([PR.norm(x) == PR.norm(rec["pred"]) for x in sp])))
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if (n_i + 1) % 10 == 0:
                el = time.time() - t0
                log(f"  {n_i + 1}/{len(item_ids)} items, {el / 60:.1f} min elapsed")
