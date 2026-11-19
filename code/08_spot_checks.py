"""08_spot_checks.py: pick the items you check by hand and draw each one.

Plain-language summary: chooses 20 suite items by fixed rules (largest and
smallest hidden regions, X-ray box cases, random picks with a fixed seed, swaps,
text removals) and saves a picture for each one in docs/spot_checks/. Each
picture shows the question, the original image with the hidden region outlined
in orange and the control patch in green, and the swapped image. It also writes
docs/spot_checks/spot_checks.csv, the list your checklist refers to.

Run:  python code/08_spot_checks.py
"""
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from groundcheck.suite import Suite  # noqa: E402

OUT = os.path.join(ROOT, "docs", "spot_checks")
SEED = 20261119


def source_link(it):
    if it.dataset == "SLAKE":
        return ("https://huggingface.co/datasets/BoKelvin/SLAKE/blob/a9083ce6c34ac3ffb17671a605962924d8a8f9e9/test.json",
                f'search the page for "qid": {it.source_id}')
    if it.dataset == "VQA-RAD":
        row = int(it.source_id.split(":")[1])
        return (f"https://huggingface.co/datasets/flaviagiammarino/vqa-rad/viewer/default/test?row={row}",
                f"row {row} of the test split")
    if it.dataset == "PathVQA":
        row = int(it.source_id.split(":")[1])
        return (f"https://huggingface.co/datasets/flaviagiammarino/path-vqa/viewer/default/test?row={row}",
                f"row {row} of the test split")
    return ("https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA/blob/7e7c465a68eb2b866926bfa59c8c9d17a8daba65/MM/test.jsonl",
            f'search the page for "{it.source_id}"')


def pick(s: Suite):
    v = s.variants.merge(s.items, on="item_id")
    v = v[~v.item_id.str.startswith("mimic-")]  # MIMIC pictures must not land in the shared docs folder
    rng = np.random.default_rng(SEED)
    reg = v[(v.variant == "mask_region") & v.status.str.startswith("ok")].copy()
    reg["frac"] = reg.detail.map(lambda d: json.loads(d)["region_frac"])
    reg["src"] = reg.detail.map(lambda d: json.loads(d)["source"])
    chosen = []

    def add(ids, why):
        for i in ids:
            if i not in [c[0] for c in chosen]:
                chosen.append((i, why))
    add(reg.sort_values("frac", ascending=False).item_id.head(2), "extreme: largest hidden region")
    add(reg.sort_values("frac").item_id.head(2), "extreme: smallest hidden region")
    add(reg[reg.src == "bounding_box"].sample(2, random_state=SEED).item_id, "X-ray box case")
    add(reg[reg.rule == "R1_named"].sample(3, random_state=SEED).item_id, "random: named organ")
    add(reg[reg.rule != "R1_named"].sample(3, random_state=SEED).item_id, "random: abnormality")
    sw = v[(v.variant == "swap_image") & (v.status == "ok")]
    for ds in ("SLAKE", "VQA-RAD", "PathVQA", "MedXpertQA-MM"):
        add(sw[sw.dataset == ds].sample(1, random_state=SEED).item_id, f"random: swap ({ds})")
    tr = v[(v.variant == "remove_text") & (v.status == "ok")]
    add(tr.sample(3, random_state=SEED).item_id, "random: text removal")
    nr = v[(v.variant == "mask_region") & (v.status == "not_applicable") & (v.dataset == "SLAKE")]
    add(nr.sample(2, random_state=SEED).item_id, "random: SLAKE item with no region (should be a whole-image question)")
    return chosen


def outline(ax, mask, color):
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.6)


def draw(s: Suite, iid, why, n):
    it = s.item(iid)
    av = s.available(iid)
    panels = ["original"] + (["swap_image"] if "swap_image" in av else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.9), squeeze=False)
    img = Image.open(os.path.join(ROOT, it.image_path)).convert("RGB")
    ax = axes[0, 0]
    ax.imshow(img)
    ax.set_title("Original (orange = hidden, green = control)", fontsize=8.5)
    for var, col in (("mask_region", "#eb6834"), ("mask_control", "#1baf7a")):
        if var in av:
            m = np.array(Image.open(os.path.join(ROOT, s._v[(iid, var)].mask_file))).astype(bool)
            outline(ax, m, col)
    if "swap_image" in av:
        axes[0, 1].imshow(s.render(iid, "swap_image").image)
        axes[0, 1].set_title("Swapped image", fontsize=8.5)
    for a in axes[0]:
        a.set_xticks([]); a.set_yticks([])
    det = s._v[(iid, "mask_region")]
    q = it.question if len(it.question) < 300 else it.question[:300] + "..."
    txt = f"#{n} {iid}  [{why}]\nQ: {q}\nA: {it.answer}"
    if det.status.startswith("ok"):
        txt += f"\nHidden: {json.loads(det.detail)['labels']} (rule {det.rule})"
    fig.text(0.01, 0.01, "\n".join(textwrap.fill(l, 110) for l in txt.split("\n")), fontsize=7.5, va="bottom")
    fig.subplots_adjust(bottom=0.27, top=0.93, wspace=0.05)
    p = os.path.join(OUT, f"{n:02d}_{iid}.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return os.path.relpath(p, ROOT)


def main():
    import shutil
    shutil.rmtree(OUT, ignore_errors=True)  # no pictures left over from an older selection
    os.makedirs(OUT, exist_ok=True)
    s = Suite(ROOT)
    rows = []
    for n, (iid, why) in enumerate(pick(s), 1):
        it = s.item(iid)
        link, how = source_link(it)
        tr = s._v[(iid, "remove_text")]
        rows.append(dict(n=n, item_id=iid, why=why, dataset=it.dataset, source_id=it.source_id,
                         answer=it.answer, picture=draw(s, iid, why, n), source_link=link, how_to_find=how,
                         text_after_removal=tr.text if tr.status == "ok" else ""))
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "spot_checks.csv"), index=False)
    print(f"{len(rows)} spot checks written to docs/spot_checks/")


if __name__ == "__main__":
    main()
