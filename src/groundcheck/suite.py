"""Load the Evidence Perturbation Suite and render any item under any variant.

Plain-language summary: the suite stores recipes (masks, swap choices, edited
text), not thousands of altered image copies. This file turns a recipe back into
the exact image and question a model sees. The same recipe always gives the same
picture.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

from . import perturb as P

VARIANTS = ("original", "mask_region", "mask_control", "swap_image", "remove_text")
EVIDENCE_REMOVING = ("mask_region", "swap_image", "remove_text")
PRIVATE_DATASETS = ("MIMIC-CXR-VQA",)  # outputs on these items never leave the author's machine


def is_private(item_id: str) -> bool:
    return item_id.startswith("mimic-")


@dataclass
class Rendered:
    item_id: str
    variant: str
    image: Image.Image
    question: str


class Suite:
    def __init__(self, root: str):
        self.root = root
        parts = [os.path.join(root, "data", "suite")]
        priv = os.path.join(root, "data", "suite_private")
        if os.path.exists(os.path.join(priv, "items.csv")):
            parts.append(priv)  # MIMIC-CXR, credentialed: present only on the author's machine
        rd = lambda d, f: pd.read_csv(os.path.join(d, f), keep_default_na=False, dtype=str)
        self.items = pd.concat([rd(d, "items.csv") for d in parts], ignore_index=True)
        self.variants = pd.concat([rd(d, "variants.csv") for d in parts], ignore_index=True)
        self._v = {(r.item_id, r.variant): r for r in self.variants.itertuples(index=False)}
        self._i = {r.item_id: r for r in self.items.itertuples(index=False)}

    def item(self, item_id: str):
        return self._i[item_id]

    def available(self, item_id: str) -> list[str]:
        return [v for v in VARIANTS if self._v[(item_id, v)].status.startswith("ok")]

    def render(self, item_id: str, variant: str) -> Rendered:
        it = self._i[item_id]
        rec = self._v[(item_id, variant)]
        if not rec.status.startswith("ok"):
            raise ValueError(f"{item_id}/{variant} is {rec.status}")
        img = Image.open(os.path.join(self.root, it.image_path)).convert("RGB")
        q = it.question
        if variant in ("mask_region", "mask_control"):
            region = np.array(Image.open(os.path.join(self.root, rec.mask_file))).astype(bool)
            img = P.apply_region(img, region)
        elif variant == "swap_image":
            img = Image.open(os.path.join(self.root, rec.swap_image_path)).convert("RGB")
        elif variant == "remove_text":
            q = rec.text
        return Rendered(item_id, variant, img, q)
