# Build spec

*Plain-language header: the one-page plan the whole build follows. Each line says what one design choice is. The right-hand notes say what it means for you. You do not need to edit this; tell Claude if anything is wrong.*

```
PROJECT:        GroundCheck (evidence-grounded confidence) + Evidence Perturbation Suite.
                Archetypes: benchmark (the suite) + pipeline/tool (the method) + analysis paper (results).
QUESTION:       Does a clinical vision-language model's confidence in an answer survive when the
                evidence behind it is hidden, swapped or removed, and does lowering confidence that
                survives make the model's confidence more honest?
SOURCES:        SLAKE en test, HF BoKelvin/SLAKE @a9083ce, CC BY 4.0, accessed 2026-11-19
                VQA-RAD test, HF flaviagiammarino/vqa-rad @bcf91e7, CC0 1.0, accessed 2026-11-19
                PathVQA test, HF flaviagiammarino/path-vqa @1685832, MIT, accessed 2026-11-19
                MedXpertQA MM test (single-image), HF TsinghuaC3I/MedXpertQA @7e7c465, MIT, accessed 2026-11-19
                MIMIC-Ext-MIMIC-CXR-VQA v1.0.0 test + MIMIC-CXR-JPG v2.1.0 images, PhysioNet, credentialed
                (PhysioNet Credentialed Health Data License 1.5.0); built locally, never redistributed
UNIT:           one question about one image (an "item"); each item has up to 5 variants.
MEASURES:       Baselines: token probability; verbalized confidence; sampling consistency (k=5, T=1.0);
                split-conformal prediction sets (alpha 0.1, closed-set items); Platt scaling.
                Survival u_v = (p_v(a0) - chance) / (p_ref(a0) - chance), clipped to [0,1], where a0 is
                the model's original answer, p_v its probability under variant v, and p_ref the original
                (or the control patch, for region hiding).
                GroundCheck = c0 * (1 - lambda * mean_v u_v), lambda fit per dataset on the calibration split.
                GroundCheck-logistic = logistic model on [logit c0, u_v, missing flags], fit on calibration split.
                Flag = c0 >= 0.8 and u >= 0.8.
                Metrics on the test split: ECE (15 bins), Brier, AUROC, AURC, bootstrap 95% CIs.
OUTPUTS:        data/suite/{items.csv, variants.csv, masks/, qa_report.txt}; docs/CODEBOOK.md;
                results/<run>/{raw.jsonl, item_scores.csv, metrics.json}; paper/stats.json;
                paper/tables/metrics_table.{csv,md}; paper/figures/fig1..fig3.
VENUES:         GitHub repo -> Zenodo record 1 (Suite v1.0, dataset) -> Zenodo record 2 (method code v1.0,
                software) -> results with the method release -> preprint (arXiv cs.CV or cs.LG; author submits).
VERIFY POINTS:  decisions D1-D9 in docs/VERIFY_CHECKLIST.md Part 4; spot checks of masks and swaps by eye;
                reproduce suite counts; confirm licenses on source pages.
LICENSE:        code MIT; suite CC BY 4.0 (sources keep their own licenses; see LICENSE-DATA.md).
DECIDED:        name GroundCheck; datasets SLAKE, VQA-RAD, PathVQA, MIMIC-CXR, MedXpertQA; code MIT;
                author line as in AUTHORS.json (2026-11-19).
ASSUMPTIONS:    model list (three matched general/medical pairs); suite license CC BY 4.0.
                Calibration/test split is 50/50 by image, seeded, so one image never sits in both halves.
```

| Spec line | What it means for you |
|---|---|
| Unit | Every row in `items.csv` is one question about one image. |
| Survival | A number from 0 to 1. 0 means the model's confidence fell to guessing level once the evidence was gone. 1 means it did not fall at all. |
| GroundCheck | The method's adjusted confidence: original confidence, lowered in proportion to how much survived. |
| Calibration / test split | The method is tuned on one half of the questions and graded on the other half, so it never grades itself on its own tuning data. |
| Control patch | A same-size patch somewhere else on the body. It separates "hiding the evidence" from "the image has a patch on it". |
