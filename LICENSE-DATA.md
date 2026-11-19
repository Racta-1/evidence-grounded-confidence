# Data license

The Evidence Perturbation Suite files created by this project (`data/suite/items.csv`,
`data/suite/variants.csv`, `data/suite/masks/`, the QA report, and the codebook) are
released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**,
https://creativecommons.org/licenses/by/4.0/ . Credit: Taye Akinrele, Evidence Perturbation Suite.

The suite is built from three public datasets that keep their own licenses. The suite
stores recipes (masks, swap choices, edited question text); the original images are
downloaded from their sources by `code/01_fetch.py`.

| Source | License (as stated on the Hugging Face dataset card) | Cite |
|---|---|---|
| SLAKE (English subset, test split) | CC BY 4.0 | Liu B, et al. SLAKE. IEEE ISBI 2021. |
| VQA-RAD (test split) | CC0 1.0 | Lau JJ, et al. Scientific Data 5, 180251 (2018). |
| MedXpertQA MM (test split, single-image items) | MIT | Zuo Y, et al. MedXpertQA. arXiv:2501.18362 (2025). |

Edited question text in `variants.csv` (`remove_text` rows) is derived from MedXpertQA
and is redistributed under its MIT license with attribution.
