# Codebook: Evidence Perturbation Suite and results files

*Plain-language header: every column in every table this project produces, what it means, and where it comes from. Open this when a column name in a spreadsheet is unclear.*

## `data/suite/items.csv`: one row per question

| Column | Meaning | Source / transformation |
|---|---|---|
| `item_id` | Unique ID. `slake-<qid>`, `vqarad-test-<row>`, `pathvqa-test-<row>`, `mxq-<id>`, `mimic-<idx>` | Built from the source ID |
| `dataset` | `SLAKE`, `VQA-RAD`, `PathVQA`, `MedXpertQA-MM`, or `MIMIC-CXR-VQA` (private) | |
| `source_id` | ID in the original dataset (SLAKE `qid`; VQA-RAD `test:<row index>` in the pinned parquet; MedXpertQA `id`) | Source |
| `split` | `cal` (tuning half) or `test` (grading half) | Seeded hash of the image, so all questions on one image share a split |
| `image_key` | Image identifier (SLAKE folder `xmlabN`; VQA-RAD first 12 hex of the image's SHA-256; MedXpertQA file name) | |
| `image_path` | Path to the image after running `code/01_fetch.py` (VQA-RAD images are extracted to `data/interim/`) | |
| `question` | Question text, unchanged | Source |
| `answer` | Correct answer, unchanged (MedXpertQA: the letter) | Source |
| `answer_type` | `closed` (yes/no), `choice` (pick one of the options named in an "X or Y" question; the correct answer must be exactly one of them), `open` (free text), `mcq` (letter A to E) | SLAKE `answer_type`; VQA-RAD: yes/no answers are closed; MedXpertQA: mcq |
| `options` | JSON list of choices for closed/mcq items | |
| `content_type` | SLAKE question category (Organ, Position, Abnormality, ...); MedXpertQA medical task | Source |
| `modality` | CT / MRI / X-Ray (SLAKE only) | Source |
| `location` | Body region (SLAKE) or body system (MedXpertQA) | Source |

## `data/suite/variants.csv`: one row per question per variant (5 per question)

| Column | Meaning |
|---|---|
| `item_id` | Links to `items.csv` |
| `variant` | `original`, `mask_region`, `mask_control`, `swap_image`, `remove_text` |
| `status` | `ok` (made), `not_applicable` (this variant does not apply to this question), `unavailable` (applies but could not be made; reason in `detail`) |
| `rule` | Which rule produced the variant (table below) |
| `detail` | JSON with specifics: hidden `labels`, mask `source` (pixel_mask or bounding_box), `region_frac` (share of the image hidden), `swap_image_key`, `chars_removed` |
| `mask_file` | Black-and-white PNG, same size as the image; white = painted over |
| `swap_image_path` | Image shown instead of the original |
| `text` | Question text shown for `remove_text` |

### Rules

| Rule | Plain meaning |
|---|---|
| `R1_named` | The question or answer names an outlined organ; hide it (and lesions annotated inside it, such as Liver Cancer inside Liver). "Left lung" hides only the left lung; "lung" hides both. |
| `R2_abnormality` | The question asks about an abnormality; hide every annotated lesion on that image (cancer, tumor, edema, nodule, mass, effusion, pneumonia, cardiomegaly, atelectasis, pneumothorax, infiltrate). |
| `R1+R2` | Both of the above. |
| `R0_none` | Nothing on the image is referenced (for example "What modality is this?"); no region variant. |
| Region is grown by 3% of image width; painted with the median gray of the body (decisions D1, D2). |
| Chest X-rays in SLAKE have no organ outlines (their `mask.png` is a plain picture), so they use the dataset's lesion boxes. |
| `mask_control` | Same shape and size as the hidden region, moved to a seeded random spot that does not touch it and lies at least half on the body (D3). |
| `S1_same_modality_other_region` | SLAKE: swapped image has the same modality (CT/MRI/X-ray) but a different body region. |
| `S1b_other_region_any_modality` | SLAKE: no same-modality image of another region exists (all SLAKE X-rays are chest), so a different body region of any modality is used. |
| `S2_random_other_image` | VQA-RAD: a different image, chosen at random (VQA-RAD has no body-region labels). |
| `S2_random_other_image` (PathVQA) | Same as VQA-RAD: a different image chosen at random (no organ or body-region labels). |
| `S5_unrelated_nonchest_slake` | MIMIC-CXR: an unrelated non-chest CT or MRI image from SLAKE (decision D10). |
| `S3_other_body_system` | MedXpertQA: an image from a case in a different body system. |
| `T1_final_question_only` | MedXpertQA: the patient story is removed; only the final question sentence and the answer choices remain. |
| `T0_no_context` | Nothing separable to remove (the question is one sentence). |

## Private files (MIMIC-CXR, on the credentialed user's machine only)

`data/suite_private/items.csv` and `variants.csv` use the same columns as above; `qa_report.txt` and `skipped.txt` describe them. Only MIMIC questions with exactly one answer are kept (D11). The split is by patient. Model outputs on these items are written to `results/<run>/raw_private.jsonl` and `item_scores_private.csv`. Git and the release packager exclude all of them.

## `results/<run>/raw.jsonl`: one line per question per variant shown to the model

`item_id`, `variant`, `model`, `answer_type`; closed/mcq: `probs` (each option's probability), `pred`; open: `gen_text`, `gen_mean_lp` (mean log-probability per token), `pred`; all: `p_a0` (probability of the model's ORIGINAL answer under this variant). Original rows add `a0`, `correct`, `verbal_text`, `verbal_conf`, `samples`, `consistency`.

## `results/<run>/item_scores.csv`: one row per question

| Column | Meaning |
|---|---|
| `correct` | The model's original answer matched the correct answer (closed/mcq: exact; open: normalized exact or whole-word containment) |
| `chance` | Guessing level: 1/2 for yes/no, 1/5 for A to E, 0 for open |
| `conf_token` | Token-probability confidence in the original answer |
| `conf_verbal` | The number the model stated ("Confidence: 80" gives 0.80); empty if it did not state one |
| `conf_consistency` | Share of 5 sampled answers that agree with the main answer |
| `p_a0__<variant>` | Probability of the original answer under each variant |
| `changed__<variant>` | The model's answer changed under that variant |
| `u__<variant>` | Survival: share of above-chance confidence that remained (0 to 1) |
| `u` | Mean survival over the variants that remove evidence (decision D6) |
| `conf_platt` | Token confidence after Platt recalibration (fit on `cal`) |
| `conf_groundcheck` | **The method's adjusted confidence** |
| `conf_groundcheck_logistic` | The learned variant of the method |
| `flag` | Confident (at least 0.8) and kept at least 80% of that confidence without its evidence (D7) |

## `paper/stats.json`

Every number any document quotes. Generated by `code/05_score.py`; never edited by hand.
