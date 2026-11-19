# Publish guide

*Plain-language header: step-by-step instructions for putting the project online and getting its two permanent citation numbers (DOIs). Do this only after you finish `docs/VERIFY_CHECKLIST.md` and approve publishing. Total time: about 30 minutes. Nothing here has been published yet: Claude had no GitHub or Zenodo access in this session, so every step is yours or can be done by Claude in a later session with a token you provide.*

## Copy-ready text

**Repository name:** `evidence-grounded-confidence`

**Repository description:** Evidence-grounded confidence for clinical vision-language models: does a model's confidence survive when its image evidence is hidden, swapped, or removed?

**Author (both records):** Akinrele, Taye | ORCID 0009-0003-0564-9001 | Independent Researcher, Tuscaloosa, Alabama, USA

**Record 1 title:** Evidence Perturbation Suite v1.0.0: masked-region, swapped-image, and removed-text variants for medical visual question answering
**Record 1 type / license:** Dataset / Creative Commons Attribution 4.0
**Record 1 keywords:** medical visual question answering; uncertainty estimation; evidence perturbation; vision-language models; calibration; benchmark; SLAKE; VQA-RAD; MedXpertQA

**Record 2 title:** EGC: Evidence-Grounded Confidence for clinical vision-language models, code and configurations v1.0.0
**Record 2 type / license:** Software / MIT
**Record 2 keywords:** uncertainty estimation; calibration; vision-language models; clinical AI; reliability; conformal prediction; evidence grounding

Descriptions: copy the `description` field from `dist/zenodo_suite.json` and `dist/zenodo_method.json`, after you rewrite them in your own words (checklist Part 5).

## Step 1: Put the project on GitHub (10 min)

**Fastest: let Claude do it.** Create a token at https://github.com/settings/tokens/new. A token is a temporary password that lets Claude upload for you, and you can cancel it at any time. Give it the note "egc upload", set expiration to 7 days, tick the **repo** box, click **Generate token**, and copy it. In a Claude session, say "here is my GitHub token for the upload" and paste it. Claude runs `tools/publish_github.py`, which checks the publish gate first. Afterward, delete the token on the same page.

**Or do it yourself:**
1. Go to https://github.com/new. Repository name: `evidence-grounded-confidence`. Choose **Public**. Leave "Add a README" **unticked** (the project has one). Click **Create repository**.
2. In a terminal inside the project folder, paste:
   ```
   git remote add origin https://github.com/Racta-1/evidence-grounded-confidence.git
   git push -u origin main
   ```
   (The project already has its first commit.) Refresh the GitHub page. You should see the README displayed under the file list.

## Step 2: Record 1, the Suite, on Zenodo (10 min)

1. Log in at https://zenodo.org using **Log in with ORCID**, so your ORCID attaches automatically.
2. Click **New upload** (top right).
3. Drag in `dist/egc-perturbation-suite-v1.0.0.zip`. Claude builds it with `python code/07_package.py --version 1.0.0` after your approval.
4. Fill in: **Resource type** Dataset. **Title**, **Creators**, **Description**, **License** and **Keywords** from the copy-ready text above. **Version** 1.0.0. Under **Related works**, add the GitHub link with relation "Is supplemented by".
5. Click **Preview** and check that your name and ORCID show correctly. Then click **Publish**. The screen shows a DOI like `10.5281/zenodo.1234567`. Copy it.

## Step 3: Record 2, the method code, on Zenodo (10 min)

Same steps with `dist/egc-method-v1.0.0.zip` and the Record 2 text. **Resource type** Software. Under **Related works**, add Record 1's DOI with relation "Requires". Your plan times this record for after the full model results (May 2027), so the results ship inside it.

*Alternative for later versions:* Zenodo, then the **GitHub** tab, then flip the switch for `evidence-grounded-confidence`. Every new GitHub release then mints a new DOI automatically.

## Step 4: Record what happened, the same day (5 min)

Add a row to `docs/evidence_log.csv` for each record: date, title, DOI, link. Save a PDF of each Zenodo page (browser **Print > Save as PDF**). Then send Claude the DOIs, and Claude puts them into the README, `CITATION.cff` and START_HERE. Records not logged the day they happen are the ones that go missing later.
