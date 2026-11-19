# GroundCheck: Evidence-Grounded Confidence for Clinical Vision-Language Models

**Status:** DRAFT v0.1.0 (unverified; not for citation) | **Maintainer:** Taye Akinrele, [ORCID 0009-0003-0564-9001](https://orcid.org/0009-0003-0564-9001) | **License:** code MIT, suite CC BY 4.0 ([LICENSE-DATA.md](LICENSE-DATA.md))

**In plain terms:** Medical AI models that read images also report how sure they are. This project checks whether that confidence actually rests on the image and the case details. It hides the part of the scan the answer depends on, swaps in an unrelated scan, or removes the patient history, and watches whether confidence falls. Answers that stay confident with their evidence gone are flagged, and their reported confidence is lowered.

New here? Read [docs/START_HERE.md](docs/START_HERE.md) first. Terms are explained in [docs/GLOSSARY.md](docs/GLOSSARY.md).

This repository holds two releases. The **Evidence Perturbation Suite** covers 9,812 public questions from SLAKE, VQA-RAD, PathVQA and MedXpertQA (multimodal), plus MIMIC-CXR questions that each credentialed user builds locally (see below). It provides hidden-region, control-patch, swapped-image and removed-text variants, with masks, codebook, QA report and provenance log. The **GroundCheck method** measures how much above-chance confidence in a model's original answer survives each evidence-removing perturbation. It flags confident answers whose confidence survives, and adjusts confidence as `c0 * (1 - lambda * survival)`, with lambda fit on a held-out calibration split. It is compared against token probability, verbalized confidence, sampling consistency, split-conformal prediction and Platt scaling. Configurations cover three matched general/medical open-weight pairs: Qwen2.5-VL-7B / Lingshu-7B, Gemma-3-4B / MedGemma-4B, and LLaVA-1.6-Mistral-7B / LLaVA-Med-1.5-Mistral-7B.

## What is here

```
code/        numbered scripts in run order (01 fetch ... 07 package)
src/groundcheck/     the library: perturb, suite, models, prompts, method, metrics, score, run
configs/     one file per model (six GPU models + a CPU pilot)
data/raw/    PROVENANCE.txt (raw files are re-downloaded by code/01_fetch.py, pinned versions)
data/suite/  the Evidence Perturbation Suite: items.csv, variants.csv, masks/, qa_report.txt
results/     one folder per model run
paper/       stats.json (every quoted number), figures/, tables/
docs/        START_HERE, GLOSSARY, BUILD_SPEC, CODEBOOK, LIMITATIONS, VERIFY_CHECKLIST,
             GPU_RUN_GUIDE, NEXT_STEPS, PUBLISH_GUIDE, TALK_TRACK
tests/       automatic tests (python -m pytest -q tests)
tools/       publish gate and upload helpers
```

## Run it

```bash
pip install -r requirements.txt
python code/01_fetch.py            # download pinned public sources (about 910 MB), log provenance
python code/01b_fetch_mimic.py --user YOUR_PHYSIONET_USERNAME   # optional; credentialed PhysioNet access
python code/02_build_suite.py      # build the suite; ends with 9 PASS lines (another 8 for MIMIC if fetched)
python code/03_preview_suite.py    # example sheet: paper/figures/fig1_suite_examples.png
python -m pytest -q tests          # 18 tests
python code/04_run_model.py configs/qwen2_5_vl_7b.yaml     # GPU; repeat per config
python code/05_score.py            # per-item scores, metrics, paper/stats.json
python code/06_figures.py          # figures
```
CPU-only check of the whole pipeline (first `pip install num2words`): `python code/04_run_model.py configs/pilot_smolvlm_cpu.yaml` (about 40 minutes on 2 cores). GPU instructions: [docs/GPU_RUN_GUIDE.md](docs/GPU_RUN_GUIDE.md).

## Sources

| Source | Version (pinned) | License (dataset card) | Accessed |
|---|---|---|---|
| SLAKE, English test split. Liu et al., ISBI 2021. https://huggingface.co/datasets/BoKelvin/SLAKE | commit a9083ce | CC BY 4.0 | 2026-11-19 |
| VQA-RAD, test split. Lau et al., Sci Data 2018. https://huggingface.co/datasets/flaviagiammarino/vqa-rad | commit bcf91e7 | CC0 1.0 | 2026-11-19 |
| PathVQA, test split. He et al., arXiv:2003.10286. https://huggingface.co/datasets/flaviagiammarino/path-vqa | commit 1685832 | MIT | 2026-11-19 |
| MedXpertQA MM, test split, single-image items. Zuo et al., arXiv:2501.18362. https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA | commit 7e7c465 | MIT | 2026-11-19 |

## Headline numbers (from `data/suite/qa_report.txt`; do not cite until verified)

The public suite holds 9,812 questions: SLAKE 1,061, VQA-RAD 451, PathVQA 6,719, MedXpertQA 1,581. It defines 392 hidden-region variants, 353 control patches, 9,812 swapped images and 1,260 removed-text variants. Model results are pending the GPU runs. The CPU pilot in `results/pilot_smolvlm256m_cpu/` checks the pipeline only.

## MIMIC-CXR (credentialed)

MIMIC-CXR questions come from MIMIC-Ext-MIMIC-CXR-VQA v1.0.0 with images from MIMIC-CXR-JPG v2.1.0, both on PhysioNet under the PhysioNet Credentialed Health Data License. Nothing from MIMIC is in this repository or its releases. Each credentialed user runs `code/01b_fetch_mimic.py`, which builds the MIMIC items into `data/suite_private/`. Model outputs on MIMIC items go to `results/<run>/raw_private.jsonl` and `item_scores_private.csv`. All of these are excluded by `.gitignore` and by the release packager. Only summary metrics are shared.

## Limitations

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) before using or citing anything here.

## AI assistance

Claude (Anthropic) assisted with code and data handling. The author made the design decisions and checks every score against source output (see `docs/VERIFY_CHECKLIST.md`).

## Citation

See [CITATION.cff](CITATION.cff). DOI: pending first release.
