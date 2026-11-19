"""06_figures.py: draw the results figures from the scored runs.

Plain-language summary: two charts per model run.
  fig2_<run>_confidence_drop.png  How much confidence in the same answer remains
                                  when the evidence is hidden, swapped or removed.
  fig3_<run>_reliability.png      Stated confidence vs actual accuracy, before and
                                  after the evidence-grounded adjustment (the
                                  closer to the diagonal, the more honest).
Every figure says DRAFT until you run with --final after verification.

Run:  python code/06_figures.py [--final]
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from egc import metrics as MT  # noqa: E402

# reference categorical palette (dataviz skill), fixed slot order
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
LABEL = {"original": "Original", "mask_control": "Control patch", "mask_region": "Region hidden",
         "swap_image": "Image swapped", "remove_text": "Text removed"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})


def stamp(fig, final):
    if not final:
        fig.text(0.995, 0.995, "DRAFT", ha="right", va="top", fontsize=13, color="#b00020", weight="bold")


def fig_drop(df, run, out, final, title_note):
    ds_list = [d for d in ["SLAKE", "VQA-RAD", "MedXpertQA-MM"] if d in set(df.dataset)]
    order = ["original", "mask_control", "mask_region", "swap_image", "remove_text"]
    ypos = {v: len(order) - 1 - i for i, v in enumerate(order)}  # same row for a variant in every panel
    fig, axes = plt.subplots(1, len(ds_list), figsize=(4.2 * len(ds_list), 3.9), squeeze=False)
    for k, (ax, ds) in enumerate(zip(axes[0], ds_list)):
        d = df[df.dataset == ds]
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for v in order:
            x = d[f"p_a0__{v}"].dropna().values
            if len(x) == 0:
                continue
            rng = np.random.default_rng(0)
            bs = [rng.choice(x, len(x)).mean() for _ in range(500)]
            c = BLUE if v == "original" else (AQUA if v == "mask_control" else ORANGE)
            y = ypos[v]
            ax.hlines(y, np.quantile(bs, 0.025), np.quantile(bs, 0.975), color=c, lw=2)
            ax.scatter([x.mean()], [y], color=c, s=60, zorder=3, edgecolor="white", linewidth=2)
            ax.annotate(f"{x.mean():.2f}  (n={len(x)})", (x.mean(), y), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.set_yticks([ypos[v] for v in order])
        ax.set_yticklabels([LABEL[v] for v in order] if k == 0 else [])
        ax.set_ylim(-0.6, len(order) - 0.3)
        ax.set_xlim(0, 1.05)
        ax.set_title(ds, fontsize=11, color=INK, pad=10)
        ax.set_xlabel("Mean confidence in the original answer")
        ax.tick_params(axis="y", length=0)
    fig.suptitle(f"Does confidence fall when the evidence is gone?  {title_note}", fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Bars: 95% bootstrap intervals. Empty rows: variant not available for that dataset.",
             fontsize=8, color=MUTED)
    stamp(fig, final)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    p = os.path.join(out, f"fig2_{run}_confidence_drop.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_reliability(df, run, out, final, title_note):
    t = df[df.split == "test"]
    ds_list = [d for d in ["SLAKE", "VQA-RAD", "MedXpertQA-MM"] if d in set(t.dataset)]
    fig, axes = plt.subplots(1, len(ds_list), figsize=(3.8 * len(ds_list), 3.8), squeeze=False)
    for ax, ds in zip(axes[0], ds_list):
        d = t[t.dataset == ds]
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)))
        for col, name, c, mk in [("conf_token", "Token probability", BLUE, "o"),
                                 ("conf_egc", "Evidence-adjusted (EGC)", ORANGE, "s")]:
            b = [r for r in MT.reliability_bins(d[col], d.correct, 5) if r["n"] > 0]
            ece = MT.ece(d[col], d.correct)
            ax.plot([r["conf"] for r in b], [r["acc"] for r in b], color=c, lw=2, marker=mk, ms=7,
                    markeredgecolor="white", markeredgewidth=1.5, label=f"{name} (ECE {ece:.2f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_title(f"{ds} (test, n={len(d)})\nECE: token {MT.ece(d.conf_token, d.correct):.2f}, "
                     f"EGC {MT.ece(d.conf_egc, d.correct):.2f}", fontsize=10, color=INK)
    fig.suptitle(f"Reliability before and after the evidence adjustment.  {title_note}", fontsize=11, x=0.01, ha="left")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, [x.split(" (ECE")[0] for x in l], loc="lower center", ncol=2, frameon=False, fontsize=9)
    stamp(fig, final)
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    p = os.path.join(out, f"fig3_{run}_reliability.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    out = os.path.join(ROOT, "paper", "figures")
    os.makedirs(out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "*", "item_scores.csv"))):
        run = os.path.basename(os.path.dirname(f))
        meta = json.load(open(os.path.join(os.path.dirname(f), "run_meta.json")))
        df = pd.read_csv(f)
        note = f"Model: {meta['config']['model_id'].split('/')[-1]}"
        if "pilot" in run:
            note += " (CPU pilot, pipeline check only)"
        print(fig_drop(df, run, out, a.final, note))
        print(fig_reliability(df, run, out, a.final, note))


if __name__ == "__main__":
    main()
