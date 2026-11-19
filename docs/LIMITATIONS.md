# Limitations

*Plain-language header: an honest list of what this project cannot tell you. Read it before citing anything. Every credible project has one.*

1. **Region hiding exists only for SLAKE, and only for 392 of its 1,061 questions.** The other datasets have no organ outlines. Questions about the whole image ("What modality is this?") have no single region to hide.
2. **Chest X-rays use boxes, not outlines.** SLAKE's X-ray "mask" files are plain pictures, so X-ray items hide the dataset's lesion boxes. Boxes also cover nearby normal tissue.
3. **Hiding a region may not remove all the evidence.** Shape, position and neighboring organs can still give away the answer. The control patch measures the effect of "any patch"; it cannot prove that hiding removed every clue.
4. **Painted regions are unnatural.** A gray blob is not something the model saw in training. Part of any confidence drop may be a reaction to the blob itself. The control patch is there to estimate this.
5. **Some confidence that survives is legitimate.** For a yes/no question, a model's prior ("most organs are normal") can be right without the image. For MedXpertQA, the patient story alone can decide the answer, so confidence that survives an image swap is not always a flaw. Averaging over the evidence types (decision D6) softens this but does not remove it.
6. **Swapped X-ray images in SLAKE cross modality.** All SLAKE X-rays are chest films, so a different body region means a CT or MRI image (361 of 1,061 swaps). A model might notice the modality change rather than the missing evidence.
7. **Text removal applies only to MedXpertQA** (1,260 of 1,581 items) and uses a simple rule: keep the last question sentence. A few questions put key facts in that last sentence, and a few refer to "the findings above".
8. **Open-ended answers are scored by a matching rule,** not by a clinician. Synonyms ("kidney" vs "renal") are counted wrong.
9. **Token probability needs access to the model's internals.** That is possible for open-weight models, not for most commercial APIs. Verbalized confidence and sampling work everywhere but are weaker signals.
10. **These are benchmark questions, not clinical deployment.** Results show how models behave on public research datasets. They do not show how they would behave on a hospital's own images or patients.
11. **The CPU pilot uses a tiny general model (SmolVLM-256M) on 90 questions.** It shows only that the pipeline works. Its numbers are not findings about clinical models.
12. **One community conversion.** The LLaVA-Med configuration uses a community conversion to the Hugging Face format, not Microsoft's original code. Its outputs should be spot-checked against the original (decision D9).
13. **License statements come from the Hugging Face dataset cards.** Confirm them on the original project pages before release (checklist Part 2).
