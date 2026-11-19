"""02_build_suite.py: build the Evidence Perturbation Suite from the raw datasets.

Plain-language summary: for every question in the three source datasets, this
script decides which altered versions can be made (hidden region, control patch,
swapped image, removed text), makes the masks, and writes two tables:
  data/suite/items.csv     one row per question
  data/suite/variants.csv  one row per question per altered version
and a quality report at data/suite/qa_report.txt. You will open the tables
and the report during verification; you do not need to read this code.

Run:  python code/02_build_suite.py
"""
import glob
import hashlib
import io
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from groundcheck import mimic as MM  # noqa: E402
from groundcheck import perturb as P  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "suite")
MASKS = os.path.join(OUT, "masks")
PRIVATE = os.path.join(ROOT, "data", "suite_private")  # MIMIC: credentialed, never pushed or released
INTERIM = os.path.join(ROOT, "data", "interim")


def split_for(group: str) -> str:
    """Calibration vs test, by image group, so one image never sits in both."""
    return "cal" if P.seed_for(group, "split") % 2 == 0 else "test"


def save_mask(region: np.ndarray, name: str) -> str:
    path = os.path.join(MASKS, name)
    Image.fromarray(region.astype(np.uint8) * 255).convert("1").save(path, optimize=True)
    return os.path.relpath(path, ROOT)


# ---------------------------------------------------------------------------
def build_slake(items, variants):
    base = os.path.join(RAW, "slake")
    legend = P.parse_mask_legend(open(os.path.join(base, "mask.txt")).read())
    qs = [q for q in json.load(open(os.path.join(base, "test.json"))) if q["q_lang"] == "en"]
    # image metadata for swaps
    meta = {}
    for q in qs:
        k = q["img_name"].split("/")[0]
        meta[k] = {"modality": q["modality"], "location": q["location"]}
    cands = [(k, m) for k, m in meta.items()]
    cache = {}
    for q in qs:
        img_key = q["img_name"].split("/")[0]
        iid = f"slake-{q['qid']}"
        img_path = os.path.join(base, "imgs", q["img_name"])
        closed = q["answer_type"] == "CLOSED"
        items.append(dict(item_id=iid, dataset="SLAKE", source_id=str(q["qid"]),
                          split=split_for("slake/" + img_key), image_key=img_key,
                          image_path=os.path.relpath(img_path, ROOT), question=q["question"],
                          answer=q["answer"], answer_type="closed" if closed else "open",
                          options=json.dumps(["Yes", "No"]) if closed and q["answer"].lower() in ("yes", "no") else "",
                          content_type=q["content_type"], modality=q["modality"], location=q["location"]))
        variants.append(dict(item_id=iid, variant="original", status="ok"))
        # --- region masks
        if img_key not in cache:
            img = Image.open(img_path)
            mpath = os.path.join(base, "imgs", img_key, "mask.png")
            marr = np.array(Image.open(mpath))[..., 0] if os.path.exists(mpath) else None
            if marr is not None and not P.is_label_mask(marr, legend):
                marr = None  # grayscale picture, not a label map (chest X-rays)
            dets = json.load(open(os.path.join(base, "imgs", img_key, "detection.json")))
            present = set()
            if marr is not None:
                present |= {legend[v] for v in np.unique(marr) if v in legend}
            for d in dets:
                present |= set(d.keys())
            gray = np.array(img.convert("L"))
            cache[img_key] = (img.size, marr, dets, sorted(present), gray)
        size, marr, dets, present, gray = cache[img_key]
        labels, rule = P.target_labels(q["question"], q["answer"], q["content_type"], present)
        if not labels:
            variants.append(dict(item_id=iid, variant="mask_region", status="not_applicable", rule=rule,
                                 detail="no annotated structure referenced by question or answer"))
            variants.append(dict(item_id=iid, variant="mask_control", status="not_applicable", rule=rule))
        else:
            region, src = P.region_from_annotations(marr, legend, dets, labels, size)
            if not region.any():
                variants.append(dict(item_id=iid, variant="mask_region", status="unavailable", rule=rule,
                                     detail=f"labels {labels} not found in mask or boxes"))
                variants.append(dict(item_id=iid, variant="mask_control", status="unavailable", rule=rule))
            else:
                region = P.dilate(region)
                frac = float(region.mean())
                mfile = save_mask(region, f"{iid}_region.png")
                variants.append(dict(item_id=iid, variant="mask_region",
                                     status="ok" if frac <= P.MAX_REGION_FRAC else "ok_flag_large",
                                     rule=rule, detail=json.dumps({"labels": labels, "source": src,
                                                                   "region_frac": round(frac, 4)}),
                                     mask_file=mfile))
                ctrl = P.control_region(region, gray, iid)
                if ctrl is None:
                    variants.append(dict(item_id=iid, variant="mask_control", status="unavailable",
                                         rule=rule, detail="no non-overlapping on-body placement found"))
                else:
                    variants.append(dict(item_id=iid, variant="mask_control", status="ok", rule=rule,
                                         detail=json.dumps({"region_frac": round(float(ctrl.mean()), 4)}),
                                         mask_file=save_mask(ctrl, f"{iid}_control.png")))
        # --- swap: same modality, different body region
        own = {"image_key": img_key, **meta[img_key]}
        sk, relaxed = P.pick_swap(iid, cands, own, must_differ=("location",), must_match=("modality",))
        variants.append(dict(item_id=iid, variant="swap_image", status="ok" if sk else "unavailable",
                             rule="S1b_other_region_any_modality" if relaxed else "S1_same_modality_other_region",
                             detail=json.dumps({"swap_image_key": sk, **({"swap_location": meta[sk]["location"],
                                                "swap_modality": meta[sk]["modality"]} if sk else {})}),
                             swap_image_path=os.path.relpath(os.path.join(base, "imgs", sk, "source.jpg"), ROOT) if sk else ""))
        variants.append(dict(item_id=iid, variant="remove_text", status="not_applicable",
                             rule="T0_no_context", detail="SLAKE questions carry no separate clinical text"))


# ---------------------------------------------------------------------------
def build_parquet_set(items, variants, name, label, prefix, pattern, rule):
    """VQA-RAD and PathVQA: image bytes inside parquet files, no region labels."""
    base = os.path.join(RAW, name)
    imgdir = os.path.join(INTERIM, f"{name}_images")
    os.makedirs(imgdir, exist_ok=True)
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, pattern)))],
                   ignore_index=True)
    keys = []
    for i, r in df.iterrows():
        b = r["image"]["bytes"]
        k = hashlib.sha256(b).hexdigest()[:12]
        p = os.path.join(imgdir, f"{k}.png")
        if not os.path.exists(p):
            Image.open(io.BytesIO(b)).convert("RGB").save(p)
        keys.append(k)
    cands = [(k, {}) for k in sorted(set(keys))]
    for i, r in df.iterrows():
        k = keys[i]
        iid = f"{prefix}-test-{i:04d}"
        ans = str(r["answer"])
        closed = ans.lower() in ("yes", "no")
        items.append(dict(item_id=iid, dataset=label, source_id=f"test:{i}", split=split_for(f"{prefix}/" + k),
                          image_key=k, image_path=os.path.relpath(os.path.join(imgdir, f"{k}.png"), ROOT),
                          question=r["question"], answer=ans, answer_type="closed" if closed else "open",
                          options=json.dumps(["yes", "no"]) if closed else "", content_type="", modality="", location=""))
        variants.append(dict(item_id=iid, variant="original", status="ok"))
        for v in ("mask_region", "mask_control"):
            variants.append(dict(item_id=iid, variant=v, status="not_applicable", rule="R0_none",
                                 detail=f"{label} has no region annotations"))
        sk, _ = P.pick_swap(iid, cands, {"image_key": k}, must_differ=(), must_match=())
        variants.append(dict(item_id=iid, variant="swap_image", status="ok" if sk else "unavailable",
                             rule=rule, detail=json.dumps({"swap_image_key": sk}),
                             swap_image_path=os.path.relpath(os.path.join(imgdir, f"{sk}.png"), ROOT) if sk else ""))
        variants.append(dict(item_id=iid, variant="remove_text", status="not_applicable", rule="T0_no_context"))


def build_vqarad(items, variants):
    build_parquet_set(items, variants, "vqa_rad", "VQA-RAD", "vqarad", "test-*.parquet", "S2_random_other_image")


def build_pathvqa(items, variants):
    build_parquet_set(items, variants, "pathvqa", "PathVQA", "pathvqa", "test-*.parquet", "S2_random_other_image")


# ---------------------------------------------------------------------------
def build_medxpertqa(items, variants):
    base = os.path.join(RAW, "medxpertqa")
    rows = [json.loads(l) for l in open(os.path.join(base, "test.jsonl"))]
    rows = [r for r in rows if len(r["images"]) == 1]  # single-image items only (decision D5)
    cands = [(r["images"][0], {"body_system": r["body_system"]}) for r in rows]
    for r in rows:
        iid = f"mxq-{r['id']}"
        img = r["images"][0]
        items.append(dict(item_id=iid, dataset="MedXpertQA-MM", source_id=r["id"], split=split_for("mxq/" + r["id"]),
                          image_key=img, image_path=os.path.relpath(os.path.join(base, "images", img), ROOT),
                          question=r["question"], answer=r["label"], answer_type="mcq",
                          options=json.dumps(sorted(r["options"].keys())), content_type=r["medical_task"],
                          modality="", location=r["body_system"]))
        variants.append(dict(item_id=iid, variant="original", status="ok"))
        for v in ("mask_region", "mask_control"):
            variants.append(dict(item_id=iid, variant=v, status="not_applicable", rule="R0_none",
                                 detail="MedXpertQA has no region annotations"))
        sk, _ = P.pick_swap(iid, cands, {"image_key": img, "body_system": r["body_system"]},
                         must_differ=("body_system",), must_match=())
        variants.append(dict(item_id=iid, variant="swap_image", status="ok" if sk else "unavailable",
                             rule="S3_other_body_system", detail=json.dumps({"swap_image_key": sk}),
                             swap_image_path=os.path.relpath(os.path.join(base, "images", sk), ROOT) if sk else ""))
        newq, removed, rule = P.remove_context(r["question"])
        variants.append(dict(item_id=iid, variant="remove_text", status="ok" if newq else "not_applicable",
                             rule=rule, detail=json.dumps({"chars_removed": removed,
                                                           "chars_original": len(P.split_stem(r["question"])[0])}),
                             text=newq or ""))


# ---------------------------------------------------------------------------
def build_mimic(items, variants) -> bool:
    """MIMIC-Ext-MIMIC-CXR-VQA test split. Runs only if code/01b_fetch_mimic.py has been run.
    Swap rule S5: an unrelated non-chest SLAKE CT/MRI image (decision D10)."""
    base = os.path.join(RAW, "mimic")
    tj = MM.find_test_json(os.path.join(base, "vqa"))
    if not tj:
        return False
    img_root = os.path.join(base, "mimic-cxr-jpg")
    slake = [q for q in json.load(open(os.path.join(RAW, "slake", "test.json"))) if q["q_lang"] == "en"]
    swap_pool = sorted({q["img_name"].split("/")[0] for q in slake if q["location"] != "Chest"})
    skipped = Counter()
    for e in MM.load_entries(tj):
        rel = MM.image_rel_path(e)
        ans = MM.normalize_answer(e.get("answer"))
        if rel is None:
            skipped["no image path"] += 1
            continue
        if ans is None:
            skipped["not a single answer"] += 1
            continue
        path = os.path.join(img_root, rel)
        if not os.path.exists(path):
            skipped["image not downloaded"] += 1
            continue
        idx = str(e.get("idx", e.get("id", len(items))))
        iid = f"mimic-{idx}"
        pid = MM.patient_id(e, rel)
        closed = ans.lower() in ("yes", "no")
        items.append(dict(item_id=iid, dataset="MIMIC-CXR-VQA", source_id=idx, split=split_for("mimic/" + pid),
                          image_key=rel, image_path=os.path.relpath(path, ROOT), question=e["question"],
                          answer=ans, answer_type="closed" if closed else "open",
                          options=json.dumps(["yes", "no"]) if closed else "",
                          content_type=str(e.get("content_type", "")), modality="X-Ray", location="Chest"))
        variants.append(dict(item_id=iid, variant="original", status="ok"))
        for v in ("mask_region", "mask_control"):
            variants.append(dict(item_id=iid, variant=v, status="not_applicable", rule="R0_none",
                                 detail="region boxes (Chest ImaGenome) not used in v1"))
        rng = np.random.default_rng(P.seed_for(iid, "swap"))
        sk = swap_pool[int(rng.integers(len(swap_pool)))]
        variants.append(dict(item_id=iid, variant="swap_image", status="ok", rule="S5_unrelated_nonchest_slake",
                             detail=json.dumps({"swap_image_key": sk}),
                             swap_image_path=os.path.relpath(os.path.join(RAW, "slake", "imgs", sk, "source.jpg"), ROOT)))
        variants.append(dict(item_id=iid, variant="remove_text", status="not_applicable", rule="T0_no_context"))
    with open(os.path.join(PRIVATE, "skipped.txt"), "w") as f:
        f.write("\n".join(f"{k}: {v}" for k, v in skipped.items()) + "\n")
    return True


# ---------------------------------------------------------------------------
def qa_report(items: pd.DataFrame, var: pd.DataFrame, private: bool = False) -> str:
    title = "PRIVATE (MIMIC-CXR, credentialed; do not share)" if private else "EVIDENCE PERTURBATION SUITE"
    L = [f"{title}: QUALITY REPORT", "=" * 44,
         "Plain-language summary: counts of questions and altered versions, how many",
         "could be made, and checks that the files are consistent. Read the 'CHECKS'",
         "section first; every line should say PASS.", ""]
    L.append("ITEMS PER DATASET AND SPLIT")
    L.append(items.groupby(["dataset", "split"]).size().unstack(fill_value=0).to_string())
    L.append("")
    L.append("ANSWER TYPES")
    L.append(items.groupby(["dataset", "answer_type"]).size().unstack(fill_value=0).to_string())
    L.append("")
    L.append("VARIANT STATUS (rows = dataset x variant)")
    m = var.merge(items[["item_id", "dataset"]], on="item_id")
    L.append(m.groupby(["dataset", "variant", "status"]).size().unstack(fill_value=0).to_string())
    L.append("")
    L.append("SWAP RULES USED")
    L.append(m[m.variant == "swap_image"].groupby(["dataset", "rule"]).size().to_string())
    L.append("")
    L.append("REGION RULES USED (SLAKE mask_region)")
    sr = m[(m.dataset == "SLAKE") & (m.variant == "mask_region")]
    if len(sr):
        L.append(sr.groupby(["rule", "status"]).size().unstack(fill_value=0).to_string())
    ok = sr[sr.status.str.startswith("ok")]
    fr = ok.detail.map(lambda d: json.loads(d)["region_frac"])
    src = ok.detail.map(lambda d: json.loads(d)["source"])
    if len(ok):
        L.append(f"region size (share of image): min {fr.min():.3f}, median {fr.median():.3f}, max {fr.max():.3f}")
        L.append("region source: " + ", ".join(f"{k}={v}" for k, v in Counter(src).items()))
    L.append("")
    tr = m[(m.dataset == "MedXpertQA-MM") & (m.variant == "remove_text") & (m.status == "ok")]
    if len(tr):
        frac = tr.detail.map(lambda d: json.loads(d)["chars_removed"] / max(1, json.loads(d)["chars_original"]))
        L.append(f"TEXT REMOVAL (MedXpertQA): share of question stem removed: median {frac.median():.2f}, "
                 f"min {frac.min():.2f}, max {frac.max():.2f}")
        L.append("")
    L.append("CHECKS")
    checks = []
    checks.append(("item_id unique", items.item_id.is_unique))
    checks.append(("one row per item per variant (5 variants)",
                   (var.groupby("item_id").variant.nunique() == 5).all() and len(var) == 5 * len(items)))
    checks.append(("every image file exists", items.image_path.map(lambda p: os.path.exists(os.path.join(ROOT, p))).all()))
    sw = var[(var.variant == "swap_image") & (var.status == "ok")]
    checks.append(("every swap image exists", sw.swap_image_path.map(lambda p: os.path.exists(os.path.join(ROOT, p))).all()))
    swm = sw.merge(items[["item_id", "image_path"]], on="item_id")
    checks.append(("no swap points to its own image", (swm.swap_image_path != swm.image_path).all()))
    mk = var[var.mask_file.fillna("") != ""]
    on_disk = set(os.listdir(MASKS))
    referenced = {os.path.basename(p) for p in var.mask_file if p}
    if not private:
        checks.append(("no leftover mask files from older builds", on_disk == referenced))
    checks.append(("every mask file exists", mk.mask_file.map(lambda p: os.path.exists(os.path.join(ROOT, p))).all()))
    # region and control never overlap
    overlap_bad = 0
    for iid, g in mk.groupby("item_id"):
        if set(g.variant) >= {"mask_region", "mask_control"}:
            a = np.array(Image.open(os.path.join(ROOT, g[g.variant == "mask_region"].mask_file.iloc[0])))
            b = np.array(Image.open(os.path.join(ROOT, g[g.variant == "mask_control"].mask_file.iloc[0])))
            overlap_bad += int((a & b).any())
    checks.append(("region and control masks never overlap", overlap_bad == 0))
    # mask size equals image size
    bad = 0
    for _, r in mk.merge(items[["item_id", "image_path"]], on="item_id").iterrows():
        if Image.open(os.path.join(ROOT, r.mask_file)).size != Image.open(os.path.join(ROOT, r.image_path)).size:
            bad += 1
    checks.append(("every mask matches its image size", bad == 0))
    for name, okk in checks:
        L.append(f"  {'PASS' if okk else 'FAIL'}  {name}")
    L.append("")
    return "\n".join(L), all(c[1] for c in checks)


def tidy(items, variants):
    it = pd.DataFrame(items)
    va = pd.DataFrame(variants)
    for c in ("rule", "detail", "mask_file", "swap_image_path", "text"):
        if c not in va:
            va[c] = ""
    va = va[["item_id", "variant", "status", "rule", "detail", "mask_file", "swap_image_path", "text"]].fillna("")
    return it, va


def main():
    import shutil
    shutil.rmtree(MASKS, ignore_errors=True)  # never leave masks from an older build behind
    os.makedirs(MASKS, exist_ok=True)
    items, variants = [], []
    build_slake(items, variants)
    build_vqarad(items, variants)
    build_pathvqa(items, variants)
    build_medxpertqa(items, variants)
    it, va = tidy(items, variants)
    it.to_csv(os.path.join(OUT, "items.csv"), index=False)
    va.to_csv(os.path.join(OUT, "variants.csv"), index=False)
    rep, passed = qa_report(it, va)
    open(os.path.join(OUT, "qa_report.txt"), "w").write(rep)
    print(rep)
    # MIMIC (private): only when the credentialed download exists on this machine
    os.makedirs(PRIVATE, exist_ok=True)
    mi, mv = [], []
    if build_mimic(mi, mv) and mi:
        pit, pva = tidy(mi, mv)
        pit.to_csv(os.path.join(PRIVATE, "items.csv"), index=False)
        pva.to_csv(os.path.join(PRIVATE, "variants.csv"), index=False)
        prep, ppass = qa_report(pit, pva, private=True)
        open(os.path.join(PRIVATE, "qa_report.txt"), "w").write(prep)
        print(prep)
        print("skipped MIMIC entries:", open(os.path.join(PRIVATE, "skipped.txt")).read())
        passed = passed and ppass
    else:
        for f in ("items.csv", "variants.csv"):
            if os.path.exists(os.path.join(PRIVATE, f)):
                os.remove(os.path.join(PRIVATE, f))
        print("MIMIC-CXR: not built (run code/01b_fetch_mimic.py first on a credentialed machine).")
    if not passed:
        sys.exit("QA FAILED: see the qa_report.txt files")


if __name__ == "__main__":
    main()
