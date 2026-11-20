"""09_answer_check.py: show model answers next to the correct answers.

Plain-language summary: a quick check that answers are being marked right or
wrong sensibly. For each public dataset it prints accuracy by question type and
sample questions with the model's raw answer, the correct answer, and the mark.
For MIMIC-CXR it prints ONLY counts and accuracy (no question text), so the
output is safe to share.

Run:  python code/09_answer_check.py qwen2_5_vl_7b
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from groundcheck import prompts as PR  # noqa: E402
from groundcheck.suite import Suite, is_private  # noqa: E402


def main():
    run = sys.argv[1]
    d = os.path.join(ROOT, "results", run)
    recs = []
    for f in ("raw.jsonl", "raw_private.jsonl"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            recs += [json.loads(l) for l in open(p)]
    s = Suite(ROOT)
    rows = []
    for r in recs:
        if r["variant"] != "original":
            continue
        it = s.item(r["item_id"])
        raw = r.get("gen_text")
        if raw is None:
            pr = r.get("probs", {})
            raw = "probs " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(pr.items(), key=lambda x: -x[1]))
        rows.append(dict(item_id=r["item_id"], dataset=it.dataset, type=r["answer_type"], gold=it.answer,
                         pred=r["pred"], raw=raw, verbal=(r.get("verbal_text") or "")[:60],
                         correct=r["correct"], private=is_private(r["item_id"])))
    df = pd.DataFrame(rows)
    out = []
    out.append(f"ANSWER CHECK for {run}: {len(df)} questions\n")
    out.append("Accuracy by dataset and question type (n, accuracy):")
    g = df.groupby(["dataset", "type"]).correct.agg(["size", "mean"]).round(2)
    out.append(g.to_string())
    for ds, sub in df[~df.private].groupby("dataset"):
        out.append(f"\n--- {ds}: sample answers ---")
        for t, st in sub.groupby("type"):
            for r in st.head(6).itertuples():
                q = s.item(r.item_id).question.replace("\n", " ")
                q = q[:90] + ("..." if len(q) > 90 else "")
                out.append(f"[{'RIGHT' if r.correct else 'wrong'}] ({t}) {r.item_id}\n"
                           f"    Q: {q}\n    correct answer: {r.gold!r}\n    model said: {r.raw[:100]!r}"
                           f"\n    marked as: {r.pred!r}\n    stated confidence text: {r.verbal!r}")
    if df.private.any():
        out.append("\n--- MIMIC-CXR-VQA: counts only (data use agreement; no question text shown) ---")
        m = df[df.private]
        out.append(f"    {len(m)} questions, accuracy {m.correct.mean():.2f}; "
                   f"answers marked empty: {(m.pred.astype(str).str.len() == 0).sum()}")
    text = "\n".join(out)
    print(text)
    with open(os.path.join(d, "answer_check.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
