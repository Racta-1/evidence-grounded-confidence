# Your verification checklist

*Plain-language version: before anything goes public under your name, you check it. Total time: about 1.5 to 2 hours, in one sitting. Tick each box and write the date. The automatic gate (`tools/publish_gate.py`) checks the mechanical items; these are yours.*

*This checklist has two rounds. **Round 1 (now)** covers the Suite and the method code, which are ready. **Round 2 (after the GPU runs)** covers the model results; Claude adds those items once the results exist.*

## Part 1: Make sure the suite can be rebuilt (15 to 30 min)

Pick one:
- [ ] **Option A, run it yourself** (needs Python 3.10 or newer). In a terminal inside the project folder, paste:
  ```
  pip install -r requirements.txt
  python code/01_fetch.py
  python code/02_build_suite.py
  python -m pytest -q tests
  ```
  You should see, near the end of the second command: `SLAKE 584 477`, `VQA-RAD 244 207`, `MedXpertQA-MM 823 758` (calibration and test counts), then **9 lines starting with PASS**. The last command should end with **16 passed**.
- [ ] **Option B, ask Claude.** In a new chat with Opus 5.5 or Fable 5, upload the project zip and say: "Re-run code/01_fetch.py and code/02_build_suite.py from scratch and give me the item counts per dataset and the number of ok variants per type." Compare them to these: 3,093 items (SLAKE 1,061; VQA-RAD 451; MedXpertQA 1,581); hidden region 392; control patch 353; swapped image 3,093; text removed 1,260.

Then:
- [ ] Open `data/raw/PROVENANCE.txt`. There should be 10 lines, one per downloaded file, each with a date, size, fingerprint (sha256) and link.

## Part 2: Check the sources and their licenses (15 min)

Open each page and confirm it loads, is free, and the license matches.
- [ ] SLAKE: https://huggingface.co/datasets/BoKelvin/SLAKE says **CC BY 4.0**. Also open the original project page, https://www.med-vqa.com/slake/, and confirm it does not state a stricter license. If it does, tell Claude: this changes what the suite may redistribute.
- [ ] VQA-RAD: https://huggingface.co/datasets/flaviagiammarino/vqa-rad says **CC0 1.0**. The original record is at https://osf.io/89kps/.
- [ ] MedXpertQA: https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA says **MIT**. Also check https://github.com/TsinghuaC3I/MedXpertQA.
- [ ] The FDA figure in START_HERE: open https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices and confirm the list's update date and count (1,614 entries, September 2026 update). If the page shows a newer count, tell Claude.

## Part 3: Check items by eye and against the source (40 to 50 min)

There are 20 pictures in `docs/spot_checks/`. The list is `docs/spot_checks/spot_checks.csv` (open it in Excel or Google Sheets). For each one:
1. Open the picture. Read the question and answer at the bottom.
2. **Does the orange outline cover what the question is about?** (For example, for "What is the largest organ?" with answer Liver, the outline should be on the liver.) Does the green control patch avoid the orange area?
3. **Is the swapped image clearly unrelated** to the question?
4. **Check the question and answer against the source.** Open the `source_link` in the CSV, follow `how_to_find`, and confirm the question and answer text match.

| # | Item | Why chosen | Region right? | Swap unrelated? | Matches source? |
|---|---|---|---|---|---|
| 1 | slake-11989 | largest hidden region | ___ | ___ | ___ |
| 2 | slake-11993 | largest hidden region | ___ | ___ | ___ |
| 3 | slake-12890 | smallest hidden region | ___ | ___ | ___ |
| 4 | slake-12853 | smallest hidden region | ___ | ___ | ___ |
| 5 | slake-12015 | X-ray box case | ___ | ___ | ___ |
| 6 | slake-12044 | X-ray box case | ___ | ___ | ___ |
| 7 | slake-12237 | named organ | ___ | ___ | ___ |
| 8 | slake-12109 | named organ | ___ | ___ | ___ |
| 9 | slake-12806 | named organ | ___ | ___ | ___ |
| 10 | slake-12701 | abnormality | ___ | ___ | ___ |
| 11 | slake-12635 | abnormality | ___ | ___ | ___ |
| 12 | slake-12628 | abnormality | ___ | ___ | ___ |
| 13 | slake-12409 | swap (SLAKE) | ___ | ___ | ___ |
| 14 | vqarad-test-0421 | swap (VQA-RAD) | n/a | ___ | ___ |
| 15 | mxq-MM-730 | swap (MedXpertQA) | n/a | ___ | ___ |
| 16 | mxq-MM-779 | text removal | n/a | ___ | ___ |
| 17 | mxq-MM-1922 | text removal | n/a | ___ | ___ |
| 18 | mxq-MM-1221 | text removal | n/a | ___ | ___ |
| 19 | slake-12087 | no region made | see note | ___ | ___ |
| 20 | slake-12279 | no region made | n/a | ___ | ___ |

For **16 to 18** (text removal), also read the `text_after_removal` column in the CSV. Is the patient story gone, with the actual question and all five choices still there?

Note on **19**: the answer is "Left Kidney", but the dataset outlined only the liver on that image. With no kidney outline, there is nothing to hide, so "no region" is the correct result. Confirm that this matches what you see.

Also open `paper/figures/fig1_suite_examples.png`. In row 4 (MRI, "Is there any abnormality in the spleen?"), does the hidden area sit on the spleen? Your eye for anatomy is the check here.

## Part 4: Your decisions (15 min)

The build made these choices provisionally. Agree or change each one. Changing D1 to D5 means re-running `code/02_build_suite.py` (2 minutes). D6 to D8 change only the scoring step.

- [ ] **D0 Method name.** Keep the working name "EGC (Evidence-Grounded Confidence)", or choose another? Recommended: decide now, because the name goes into both DOIs. Your choice: ___
- [ ] **D1 How far to grow the hidden region past the outline.** 3% of image width (recommended: covers the organ's edges) or 0% (exact outline; edges may give the organ away). Your choice: ___
- [ ] **D2 What to paint the hidden region with.** Body's median gray (recommended: least eye-catching), black (stands out, may itself change answers), or blur (keeps rough shape, removes less evidence). Your choice: ___
- [ ] **D3 Where the control patch may go.** At least half on the body (recommended), or anywhere in the image. "Anywhere" makes more controls possible (39 are currently unavailable) but some would sit on empty background, which is a weaker control. Your choice: ___
- [ ] **D4 Swaps for SLAKE chest X-rays.** All SLAKE X-rays are of the chest, so a "different body region" swap must be a CT or MRI (361 items). Keep (recommended: clearly unrelated evidence), or use a different patient's chest X-ray (a harder test, but the answer may truly be the same). Your choice: ___
- [ ] **D5 MedXpertQA items with several images** (419 items) are left out; only single-image cases are kept. Keep (recommended: not all models accept several images), or include them using the first image only. Your choice: ___
- [ ] **D6 Combining evidence types.** When a question has more than one test (for example swap and text removal), average the surviving confidence (recommended: one sufficient evidence source is not penalized as harshly), or take the maximum (stricter: any surviving confidence counts against the answer). Your choice: ___
- [ ] **D7 Flag thresholds.** Flag an answer when confidence is at least 0.80 and at least 80% of it survives without the evidence. Keep (recommended), or set other numbers. Your choice: ___
- [ ] **D8 Tuning and grading halves.** 50/50 split by image, fixed seed (recommended), or 30% tuning and 70% grading (more grading data, noisier tuning). Your choice: ___
- [ ] **D9 Model list.** Three matched pairs: Qwen2.5-VL-7B / Lingshu-7B, Gemma-3-4B / MedGemma-4B, LLaVA-1.6-Mistral-7B / LLaVA-Med-1.5-Mistral-7B. The LLaVA-Med entry uses a community conversion. Keep it with a spot check against Microsoft's original on 20 questions (recommended), drop that pair, or swap in another medical model (for example HuatuoGPT-Vision). Your choice: ___

## Part 5: Before it goes public (15 to 30 min)

- [ ] The README "In plain terms" paragraph and the Zenodo descriptions (`dist/zenodo_suite.json`, `dist/zenodo_method.json`, the `description` field) are rewritten in your own words. Nothing remains that you could not explain or defend. (Claude can do this with you line by line.)
- [ ] Your name, ORCID, affiliation and email are correct in `AUTHORS.json`.
- [ ] You have read `docs/LIMITATIONS.md` and agree with it.
- [ ] The README's "AI assistance" line is true as written.
- [ ] Draft stamps removed (Claude regenerates the figures with `--final`) and this command ends with no BLOCKED lines: `python tools/publish_gate.py . --allow-draft-in src/ code/ tools/ tests/ docs/GLOSSARY.md` (the exempted folders hold program code, where brackets are normal syntax, and the glossary's definition of a draft stamp; every document and data file is still checked).
- [ ] Evidence log row written on release day (`docs/evidence_log.csv`).

**When every box is ticked, reply to Claude in your own words that you have checked the package and approve publishing.**
