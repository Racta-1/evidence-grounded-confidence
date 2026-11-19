# Data license

The Evidence Perturbation Suite files created by this project (`data/suite/items.csv`,
`data/suite/variants.csv`, `data/suite/masks/`, the QA report, and the codebook) are
released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**,
https://creativecommons.org/licenses/by/4.0/ . Credit: Taye Akinrele, Evidence Perturbation Suite.

The public suite is built from four public datasets that keep their own licenses. The suite
stores recipes (masks, swap choices, edited question text); the original images are
downloaded from their sources by `code/01_fetch.py`.

| Source | License (as stated on the Hugging Face dataset card) | Cite |
|---|---|---|
| SLAKE (English subset, test split) | CC BY 4.0 | Liu B, et al. SLAKE. IEEE ISBI 2021. |
| VQA-RAD (test split) | CC0 1.0 | Lau JJ, et al. Scientific Data 5, 180251 (2018). |
| PathVQA (test split) | MIT | He X, et al. PathVQA. arXiv:2003.10286 (2020). |
| MedXpertQA MM (test split, single-image items) | MIT | Zuo Y, et al. MedXpertQA. arXiv:2501.18362 (2025). |

**MIMIC-CXR is not part of any release.** MIMIC-Ext-MIMIC-CXR-VQA (v1.0.0) and MIMIC-CXR-JPG (v2.1.0) are distributed by PhysioNet under the PhysioNet Credentialed Health Data License 1.5.0. Each credentialed user rebuilds the MIMIC items locally with `code/01b_fetch_mimic.py`. Only aggregate results are published.

Edited question text in `variants.csv` (`remove_text` rows) is derived from MedXpertQA
and is redistributed under its MIT license with attribution.
