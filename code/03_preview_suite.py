"""03_preview_suite.py: draw example items under each altered version.

Plain-language summary: makes one picture sheet (paper/figures/fig1_suite_examples.png)
showing real questions with their original image, the hidden region, the control
patch, and the swapped image, plus a text-removal example. Use it to eyeball
whether the hidden region really covers the structure the question asks about.

Run:  python code/03_preview_suite.py [--final]
"""
import argparse
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from egc.suite import Suite  # noqa: E402

# Fixed examples chosen to span rules: named organ (CT pixel mask), named organ
# with lesion, abnormality via X-ray bounding box, brain MRI lesion.
EXAMPLES = ["slake-12123", "slake-12786", "slake-11986", None]


def pick_examples(s: Suite):
    v = s.variants.merge(s.items, on="item_id")
    ok = v[(v.variant == "mask_control") & (v.status == "ok")]
    chosen = [e for e in EXAMPLES if e and e in set(ok.item_id)]
    for mod in ("MRI", "X-Ray", "CT"):
        if len(chosen) >= 4:
            break
        c = ok[(ok.modality == mod) & (~ok.item_id.isin(chosen))]
        if len(c):
            chosen.append(c.sort_values("item_id").item_id.iloc[0])
    return chosen[:4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    s = Suite(ROOT)
    ex = pick_examples(s)
    cols = ["original", "mask_region", "mask_control", "swap_image"]
    titles = ["Original", "Region hidden", "Control patch", "Image swapped"]
    fig, axes = plt.subplots(len(ex), 4, figsize=(11, 2.9 * len(ex) + 0.6))
    for r, iid in enumerate(ex):
        it = s.item(iid)
        for c, var in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(s.render(iid, var).image, cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=11)
        q = textwrap.fill(f"{iid} ({it.modality}): {it.question}  Answer: {it.answer}", 70)
        axes[r, 0].set_ylabel("")
        axes[r, 0].annotate(q, (0, -0.04), xycoords="axes fraction", va="top", fontsize=8)
    plt.subplots_adjust(hspace=0.45, wspace=0.04, top=0.93, bottom=0.05)
    fig.suptitle("Evidence Perturbation Suite: SLAKE examples", fontsize=13)
    if not a.final:
        fig.text(0.99, 0.995, "DRAFT", ha="right", va="top", fontsize=14, color="#b00020", weight="bold")
    out = os.path.join(ROOT, "paper", "figures")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "fig1_suite_examples.png"), dpi=130)
    # text-removal example as a small text file for the paper supplement
    v = s.variants[(s.variants.variant == "remove_text") & (s.variants.status == "ok")].sort_values("item_id")
    r0 = v.iloc[0]
    with open(os.path.join(out, "text_removal_example.txt"), "w") as f:
        f.write(f"{r0.item_id}\n\nORIGINAL QUESTION:\n{s.item(r0.item_id).question}\n\nTEXT REMOVED:\n{r0.text}\n")
    print("examples:", ex)


if __name__ == "__main__":
    main()
