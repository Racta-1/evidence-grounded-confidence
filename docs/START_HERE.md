# Start Here: Evidence-Grounded Confidence (GroundCheck) for Clinical Vision-Language Models

*Read this first. It explains your project in plain language. Technical words are explained in [GLOSSARY.md](GLOSSARY.md).*

## What you are building

A free, public test kit and method. They check whether a medical AI that reads images is confident for the right reason, meaning because of what is actually in the image and the case details. When it is not, the method lowers the confidence the AI reports.

## Why it matters

New AI models can look at a scan and answer a question about it, like "Is there an abnormality in the liver?" They also say how sure they are. A model that is wrong but sounds 95% sure can mislead a doctor. That is more dangerous than a model that says "I'm not sure."

This project tests the confidence directly. It hides the part of the image the answer depends on, swaps in an unrelated image, or removes the patient's history. Then it watches whether the model's confidence falls. If the model stays just as sure with its evidence gone, its confidence was never resting on that evidence. The method flags that answer and lowers the confidence.

An everyday comparison: a student who answers "C" with total confidence, even after you cover the question, was not reading the question.

The federal government's NIST AI Risk Management Framework lists "valid and reliable" first among the qualities of trustworthy AI. The FDA's list of AI-enabled medical devices had 1,614 entries at its September 2026 update. Developers and hospitals need practical tests like this one before they deploy these tools.

## Where the information comes from

Four sources are free and public. MIMIC-CXR needs your credentialed PhysioNet access. The project downloads exact, pinned versions and records each file in `data/raw/PROVENANCE.txt`.

| Source | In plain terms | What one row looks like | Link |
|---|---|---|---|
| SLAKE (English, test set) | 1,061 questions about 96 CT, MRI and X-ray images. Experts outlined the organs and lesions. | One image, one question ("Which organ is abnormal?"), the correct answer, and the outlines | https://huggingface.co/datasets/BoKelvin/SLAKE |
| VQA-RAD (test set) | 451 questions written by clinicians about radiology images | One image, one question, one answer | https://huggingface.co/datasets/flaviagiammarino/vqa-rad |
| PathVQA (test set) | 6,719 questions about pathology slide and textbook images | One image, one question, one answer | https://huggingface.co/datasets/flaviagiammarino/path-vqa |
| MIMIC-CXR (test set of the MIMIC-Ext-MIMIC-CXR-VQA question set) | Questions about real hospital chest X-rays. **Not public:** you download it with your own PhysioNet account, and it never leaves your machine. | One chest X-ray, one question, one answer | https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/ |
| MedXpertQA, multimodal part (test set) | 1,581 hard, board-exam-style cases. Each has a patient story, one image and five answer choices. | One case: story, image, choices A to E, correct letter | https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA |

## What you will have at the end

- [ ] **Evidence Perturbation Suite v1.0.** The test kit (the four public datasets; MIMIC-CXR items are rebuilt by each credentialed user): every question with its altered versions, the masks, a codebook and the source log. It gets its own permanent citation number (a DOI) from Zenodo.
- [ ] **Method code and configurations v1.0.** The software, with settings for six open AI models. It gets its own DOI.
- [ ] **Results across open-weight general and medical models.** Three matched pairs, each a general model and its medical version built on the same base: Qwen2.5-VL-7B vs Lingshu-7B, Gemma-3-4B vs MedGemma-4B, and LLaVA-1.6-Mistral-7B vs LLaVA-Med-1.5-Mistral-7B. These come out with the method.
- [ ] **A public project page on GitHub:** https://github.com/Racta-1/evidence-grounded-confidence (after you approve publishing)

## How it connects to your work

Your first-author study on MedLoCoMo measured how model answers change with the context each model is given. This project applies the same idea to images: change the evidence, measure the change. Your taxonomy of capability gaps in generative and agentic AI names self-monitoring as a core gap. A model that knows when its confidence is not backed by evidence is a concrete form of self-monitoring.

## What you need to do

| Your task | When | Time |
|---|---|---|
| Create Zenodo and Hugging Face accounts | Before publishing | 6 min |
| Answer the five setup questions (method name, datasets, GPU, licenses, author line) | Now | 10 min |
| Run the full model runs on a GPU, using `docs/GPU_RUN_GUIDE.md` | When you have GPU access | About 1 hour of your time, plus machine time |
| Check the results using `docs/VERIFY_CHECKLIST.md` | When the draft is ready | 1.5 to 2 hours |
| Approve publishing | After checking | 5 min |
| Create the two Zenodo records, using `docs/PUBLISH_GUIDE.md` | After approving | 20 min |

Everything else is built for you.

## The steps ahead

1. **Data collected.** Done for the four public datasets, with every file logged. MIMIC-CXR is downloaded on your cluster with your PhysioNet login.
2. **Perturbation suite built.** Done. Hidden-region, control-patch, swapped-image and removed-text versions, with 9 automatic checks passing. Now 9,812 public questions, plus your MIMIC-CXR questions once you download them.
3. **Method code written and tested.** Done. 16 automatic tests pass.
4. **CPU pilot.** Done. A tiny model ran through the whole pipeline on 90 questions to prove it works. This is not a finding about clinical models.
5. **Full runs on six models.** Needs a GPU. You run it, or ask Claude to help in a session that has one.
6. **Your check, then publish.**

*You are here: steps 1 to 4 done. Waiting on your answers to the setup questions and on GPU access for step 5.*
