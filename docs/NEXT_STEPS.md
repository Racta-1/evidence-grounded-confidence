# Next steps

*Plain-language header: what comes after this first draft, in order, so v0.1 is understood as a first step toward the v1.0 releases in your plan.*

1. **Your decisions (D0 to D9) and Round 1 verification.** Then release the **Evidence Perturbation Suite v1.0** to Zenodo (dataset DOI). The suite does not depend on any model results, so it can be released first.
2. **Full runs on the six models** (`docs/GPU_RUN_GUIDE.md`). Start with one model at `--limit 20` to confirm the setup.
3. **Round 2 verification.** Claude adds results checks to the checklist, for example "Open `results/medgemma_4b/item_scores.csv`, find slake-12237, and confirm `conf_token` against `raw.jsonl`".
4. **Method code and configurations v1.0** to Zenodo (software DOI), with the results in `results/` and `paper/`.
5. **Preprint.** Built from `paper/stats.json` so every number matches. Likely venues: arXiv (cs.CV or cs.LG; first-time posters may need an endorsement) or medRxiv's health informatics section. You submit it personally; Claude prepares the kit.
6. **Possible v1.1 additions:** multi-image MedXpertQA items; a blur fill as a second masking style; a "different patient, same region" swap as a harder test; one closed-weight model through its API using verbalized confidence and sampling only (no token probabilities).
