"""Reading MIMIC-Ext-MIMIC-CXR-VQA (credentialed; never redistributed).

Plain-language summary: finds and reads the MIMIC chest X-ray question file and
turns each usable question into a suite item. Written to tolerate small
differences in field names, because it is first run on the author's own machine.
[CHECK ON FIRST REAL RUN: field names image_path / answer / idx / subject_id.]
"""
from __future__ import annotations

import glob
import json
import os


def find_test_json(root: str) -> str | None:
    cands = [p for p in glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
             if os.path.basename(p).lower() in ("test.json", "test_dataset.json")]
    cands.sort(key=lambda p: (len(p), p))
    return cands[0] if cands else None


def load_entries(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):  # {"data": [...]} or {id: entry}
        data = data.get("data") or list(data.values())
    return data


def image_rel_path(e: dict) -> str | None:
    """Path relative to the MIMIC-CXR-JPG project root, e.g. files/p10/p10000032/s5.../x.jpg"""
    p = e.get("image_path") or e.get("image") or ""
    if not p and e.get("subject_id") and e.get("study_id") and e.get("image_id"):
        sid = str(e["subject_id"])
        p = f"p{sid[:2]}/p{sid}/s{e['study_id']}/{e['image_id']}.jpg"
    if not p:
        return None
    p = p.lstrip("/")
    i = p.find("files/")
    if i >= 0:
        return p[i:]
    return "files/" + p


def image_paths(entries: list[dict]) -> list[str]:
    return sorted({p for p in (image_rel_path(e) for e in entries) if p})


def normalize_answer(ans) -> str | None:
    """Single-answer questions only (decision D11): lists of one, or plain strings."""
    if isinstance(ans, list):
        if len(ans) != 1:
            return None
        ans = ans[0]
    if ans is None:
        return None
    ans = str(ans).strip()
    return ans or None


def patient_id(e: dict, rel: str) -> str:
    if e.get("subject_id"):
        return str(e["subject_id"])
    parts = rel.split("/")
    return parts[2] if len(parts) > 2 else rel
