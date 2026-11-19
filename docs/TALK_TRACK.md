# Talking about EGC and the Evidence Perturbation Suite

*Plain-language header: what to say when someone asks about this project. Every number here matches `paper/stats.json`. It claims only what is built so far: the suite and the method. A results sentence gets added after the GPU runs and your verification.*

## 60-second version

**The problem:** AI models that read medical images also report how confident they are. A model can be confidently wrong, and a confidently wrong answer can mislead a clinician more than an honest "I'm not sure."

**What I built:** A test kit and a method that check whether that confidence rests on the evidence. For 3,093 questions from three public medical image datasets, the kit hides the part of the scan the answer depends on, swaps in an unrelated scan, or strips out the patient history. The method then measures how much of the model's confidence in the same answer survives. When a model stays confident with its evidence gone, the method flags the answer and lowers the confidence it reports.

**What it makes possible:** Developers and hospitals get a direct, repeatable test of whether a model's confidence can be trusted, run before deployment on open models they can inspect.

**Who can use it:** AI developers, hospital AI evaluation teams, regulators' and standards bodies' technical staff, and researchers.

**Where it is:** GitHub, github.com/Racta-1/evidence-grounded-confidence, with two permanent Zenodo records, one for the test kit and one for the code. (DOIs added after release.)

## Two-sentence version

I built an open test that checks whether a medical AI's confidence is grounded in the image evidence, by hiding, swapping or removing that evidence and watching whether confidence falls. Answers that stay confident without their evidence are flagged and their confidence lowered, giving developers and health systems a direct reliability check before deployment.

## Questions you may be asked

1. **Where does the data come from?** Three public research datasets: SLAKE, VQA-RAD and MedXpertQA. Each is downloaded at a pinned version, and every file is logged with its source and fingerprint. No patient data beyond what those public datasets already release.
2. **What are its limits?** Region hiding is possible only where experts outlined the anatomy (392 SLAKE questions). A painted patch is itself unusual, which is why every hidden region has a same-size control patch elsewhere. Some surviving confidence is legitimate, for example when the patient history alone decides the answer. And these are research benchmarks, not hospital data. The full list is in `docs/LIMITATIONS.md`.
3. **Why does this matter for the United States?** NIST's AI Risk Management Framework names "valid and reliable" first among the qualities of trustworthy AI. The FDA's AI-enabled device list passed 1,600 entries in 2026, and the FDA has said it intends to identify devices that use large language models. A free, open reliability test helps developers and health systems meet that bar before these tools reach patients.
