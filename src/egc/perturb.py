"""Evidence perturbations: hide the region, swap the image, remove the text.

Plain-language summary: these functions take one question about one medical
image and make the altered versions the method tests against. "Masked region"
paints over the part of the image the question is about. "Control mask" paints
over an equal-sized patch somewhere else, so we can tell "lost the evidence"
apart from "the picture just has a patch on it". "Swapped image" replaces the
picture with one from a different patient and body region. "Removed text" drops
the clinical story from a question and keeps only the final question line.

Every random choice is seeded by the item ID, so the same suite is rebuilt
exactly every time.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Settings the author rules on (see docs/VERIFY_CHECKLIST.md, decisions D1-D5)
# ---------------------------------------------------------------------------
DILATE_FRAC = 0.03          # D1: grow the hidden region by 3% of image width
FILL_MODE = "median"        # D2: paint hidden pixels with the median gray of the body (not the background)
CONTROL_MIN_BODY_FRAC = 0.5  # D3: control patch must sit at least half on the body
MAX_REGION_FRAC = 0.6       # regions larger than this share of the image are flagged

LESION_WORDS = ("cancer", "tumor", "edema", "cardiomegaly", "atelectasis", "pneumonia",
                "nodule", "mass", "effusion", "pneumothorax", "infiltrate")
ABNORMAL_TRIGGERS = ("abnormal", "lesion", "tumor", "cancer", "disease", "nodule",
                     "mass", "edema", "diseased", "unhealthy")


def seed_for(*parts: str) -> int:
    """Stable integer seed from text parts (same item -> same random choices)."""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16)


# ---------------------------------------------------------------------------
# Region targeting (SLAKE): which annotated structures does this question use?
# ---------------------------------------------------------------------------
def parse_mask_legend(text: str) -> dict[int, str]:
    legend = {}
    for line in text.splitlines():
        if ":" in line:
            v, name = line.split(":", 1)
            legend[int(v.strip())] = name.strip()
    return legend


def _norm(s: str) -> str:
    return re.sub(r"[^a-z ]+", " ", s.lower())


def _label_patterns(label: str) -> list[str]:
    """Phrases that refer to a label. 'Left Lung' matches 'left lung' and the
    generic 'lung' (generic matches are resolved below so a question about the
    left lung does not also hide the right lung)."""
    l = label.lower()
    pats = [l]
    for side in ("left ", "right "):
        if l.startswith(side):
            pats.append(l[len(side):])
    if l == "right parotid":
        pats.append("parotid")
    return pats


def target_labels(question: str, answer: str, content_type: str,
                  present: list[str]) -> tuple[list[str], str]:
    """Return (labels to hide, rule used). `present` = labels annotated on this image.

    Rules, in order:
      R1 named: the question or answer names a structure annotated on the image.
         A side-specific mention ("left lung") wins over the generic ("lung").
      R2 abnormality: the question is about an abnormality and the image has an
         annotated lesion (cancer, tumor, edema); hide the lesion(s).
      R0 none: no annotated structure is referenced; no region variant is made.
    """
    text = " " + _norm(question + " " + answer) + " "
    specific, generic = [], []
    for lab in present:
        l = lab.lower()
        if f" {l} " in text:
            specific.append(lab)
            continue
        for p in _label_patterns(lab)[1:]:
            if re.search(rf"\b{re.escape(p)}s?\b", text):
                generic.append(lab)
                break
    # drop generic matches whose specific sibling was named
    named_bases = {s.lower().replace("left ", "").replace("right ", "") for s in specific}
    generic = [g for g in generic
               if g.lower().replace("left ", "").replace("right ", "") not in named_bases]
    hits = specific + generic
    # hiding an organ also hides lesions annotated inside it ("Liver" -> "Liver Cancer")
    for h in list(hits):
        base = h.lower().replace("left ", "").replace("right ", "")
        hits += [p for p in present if p.lower().startswith(base + " ") and p not in hits
                 and any(w in p.lower() for w in LESION_WORDS)]
    lesions = [p for p in present if any(w in p.lower() for w in LESION_WORDS)]
    asks_abnormal = content_type == "Abnormality" or any(f" {w}" in text for w in ABNORMAL_TRIGGERS)
    if asks_abnormal and lesions:
        hits = sorted(set(hits) | set(lesions))
        return hits, "R2_abnormality" if not (specific or generic) else "R1+R2"
    if hits:
        return sorted(set(hits)), "R1_named"
    return [], "R0_none"


def is_label_mask(mask_arr: np.ndarray, legend: dict[int, str]) -> bool:
    """True when every pixel value is 0 or a legend code. SLAKE chest X-ray
    'mask.png' files are grayscale pictures, not label maps, and fail this test;
    for those images only the bounding boxes are used."""
    vals = set(np.unique(mask_arr).tolist())
    return vals <= ({0} | set(legend))


def region_from_annotations(mask_arr: np.ndarray | None, legend: dict[int, str],
                            detections: list[dict], labels: list[str],
                            size: tuple[int, int]) -> tuple[np.ndarray, str]:
    """Boolean HxW region for `labels`, from the pixel mask if possible, else boxes."""
    w, h = size
    region = np.zeros((h, w), bool)
    used = set()
    inv = {name: v for v, name in legend.items()}
    for lab in labels:
        hit = None
        if mask_arr is not None and lab in inv:
            hit = mask_arr == inv[lab]
        if hit is not None and hit.any():
            region |= hit
            used.add("pixel_mask")
            continue
        for det in detections:
            for name, box in det.items():
                if name == lab:
                    x, y, bw, bh = [int(round(v)) for v in box]
                    region[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)] = True
                    used.add("bounding_box")
    source = "+".join(sorted(used)) or "none"
    return region, source


def dilate(region: np.ndarray, frac: float = DILATE_FRAC) -> np.ndarray:
    r = max(1, int(round(frac * region.shape[1])))
    from scipy.ndimage import binary_dilation
    return binary_dilation(region, iterations=r)


def body_mask(gray: np.ndarray) -> np.ndarray:
    """Rough 'inside the body' mask: pixels brighter than near-black background."""
    thr = max(10, 0.5 * np.percentile(gray, 20))
    return gray > thr


def control_region(region: np.ndarray, gray: np.ndarray, item_id: str,
                   tries: int = 400) -> np.ndarray | None:
    """Same shape as `region`, moved somewhere that does not touch it and sits on the body."""
    ys, xs = np.nonzero(region)
    if len(ys) == 0:
        return None
    h, w = region.shape
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    body = body_mask(gray)
    guard = dilate(region, 0.02)
    rng = np.random.default_rng(seed_for(item_id, "control"))
    for _ in range(tries):
        dy = rng.integers(-y0, h - y1)
        dx = rng.integers(-x0, w - x1)
        if dy == 0 and dx == 0:
            continue
        moved = np.zeros_like(region)
        moved[ys + dy, xs + dx] = True
        if (moved & guard).any():
            continue
        if body[moved].mean() < CONTROL_MIN_BODY_FRAC:
            continue
        return moved
    return None


def apply_region(img: Image.Image, region: np.ndarray, mode: str = FILL_MODE) -> Image.Image:
    arr = np.array(img.convert("RGB")).copy()
    if mode == "median":
        body = body_mask(np.array(img.convert("L")))
        src = arr[body] if body.any() else arr.reshape(-1, 3)
        fill = np.median(src, axis=0).astype(arr.dtype)
        arr[region] = fill
    elif mode == "black":
        arr[region] = 0
    elif mode == "blur":
        from PIL import ImageFilter
        blurred = np.array(img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=max(4, arr.shape[1] // 40))))
        arr[region] = blurred[region]
    else:
        raise ValueError(mode)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Image swap
# ---------------------------------------------------------------------------
def pick_swap(item_id: str, candidates: list[tuple[str, dict]], own: dict,
              must_differ: tuple[str, ...], must_match: tuple[str, ...]) -> tuple[str | None, bool]:
    """Pick a replacement image key. candidates: (image_key, metadata).
    Returns (key, relaxed): relaxed=True when no candidate met `must_match`
    and the match requirement was dropped (the 'differ' requirement never is)."""
    pool = [k for k, m in candidates
            if k != own["image_key"]
            and all(m.get(f) != own.get(f) for f in must_differ)
            and all(m.get(f) == own.get(f) for f in must_match)]
    relaxed = False
    if not pool:
        relaxed = True
        pool = [k for k, m in candidates if k != own["image_key"]
                and all(m.get(f) != own.get(f) for f in must_differ)]
    if not pool:
        return None, relaxed
    pool = sorted(set(pool))
    rng = np.random.default_rng(seed_for(item_id, "swap"))
    return pool[int(rng.integers(len(pool)))], relaxed


# ---------------------------------------------------------------------------
# Text removal (MedXpertQA MM)
# ---------------------------------------------------------------------------
_SENT = re.compile(r"(?<=[.?!])\s+(?=[A-Z(])")


def split_stem(question: str) -> tuple[str, str]:
    """(stem, choices) split at 'Answer Choices:'."""
    m = re.search(r"\n?Answer Choices:", question)
    if not m:
        return question.strip(), ""
    return question[:m.start()].strip(), question[m.start():].strip()


def remove_context(question: str) -> tuple[str | None, int, str]:
    """Keep only the final question sentence plus the answer choices.

    Returns (new_question or None, characters removed, rule). None means the
    stem has no separable clinical context (a one-sentence question)."""
    stem, choices = split_stem(question)
    sents = [s.strip() for s in _SENT.split(stem.replace("\n", " ")) if s.strip()]
    if len(sents) <= 1:
        return None, 0, "T0_no_context"
    qs = [i for i, s in enumerate(sents) if s.endswith("?")]
    keep_i = qs[-1] if qs else len(sents) - 1
    kept = sents[keep_i]
    # keep a short image pointer if the kept sentence refers to a figure we dropped
    new_q = kept + ("\n" + choices if choices else "")
    removed = len(stem) - len(kept)
    if removed <= 0:
        return None, 0, "T0_no_context"
    return new_q, removed, "T1_final_question_only"
