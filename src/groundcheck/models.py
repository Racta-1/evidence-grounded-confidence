"""Model adapters: one small interface every vision-language model plugs into.

Plain-language summary: different AI models need different code to load and
query. This file hides those differences behind three questions we ask every
model: (1) how likely is each listed option, (2) what do you answer, sampled one
or more times, and (3) how likely is this specific answer. The "mock" model is a
fake used only by the automatic tests.
"""
from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class VLMAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def option_probs(self, image: Image.Image, prompt: str, options: list[str]) -> dict[str, float]:
        """Probability of each option as the answer, normalized over the options."""

    @abstractmethod
    def generate(self, image: Image.Image, prompt: str, n: int = 1, temperature: float = 0.0,
                 max_new_tokens: int = 24) -> list[tuple[str, float]]:
        """n answers with their mean token log-probability."""

    @abstractmethod
    def answer_logprob(self, image: Image.Image, prompt: str, answer: str) -> tuple[float, int]:
        """(sum log-probability, number of tokens) of `answer` given image and prompt."""


# ---------------------------------------------------------------------------
class HFAdapter(VLMAdapter):
    """Any Hugging Face image-text-to-text chat model (Qwen2/2.5-VL, LLaVA-NeXT,
    Gemma 3 / MedGemma, Idefics3 / SmolVLM, and similar)."""

    def __init__(self, model_id: str, revision: str | None = None, dtype: str = "auto",
                 device: str = "auto", max_image_side: int | None = None, processor_kwargs: dict | None = None,
                 trust_remote_code: bool = False):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.name = model_id
        tdtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision,
                                                       trust_remote_code=trust_remote_code,
                                                       **(processor_kwargs or {}))
        kw = dict(revision=revision, dtype=tdtype, trust_remote_code=trust_remote_code)
        if device == "auto":
            kw["device_map"] = "auto"
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
        if device not in ("auto",):
            self.model.to(device)
        self.tok = self.processor.tokenizer
        self.max_image_side = max_image_side

    # -- helpers ---------------------------------------------------------
    def _img(self, image):
        if self.max_image_side and max(image.size) > self.max_image_side:
            image = image.copy()
            image.thumbnail((self.max_image_side, self.max_image_side))
        return image

    def _inputs(self, image, prompt):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.processor(text=[text], images=[[self._img(image)]], return_tensors="pt")
        return inp.to(self.model.device)

    def _answer_ids(self, answer: str):
        return self.tok(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(self.model.device)

    def _forward_with(self, inp, extra_ids):
        torch = self.torch
        ids = torch.cat([inp["input_ids"], extra_ids], dim=1)
        att = torch.cat([inp["attention_mask"], torch.ones_like(extra_ids)], dim=1)
        kw = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
        if "mm_token_type_ids" in kw:  # some processors mark image tokens; extend with text type
            kw["mm_token_type_ids"] = torch.cat([kw["mm_token_type_ids"], torch.zeros_like(extra_ids)], dim=1)
        if "token_type_ids" in kw:
            kw["token_type_ids"] = torch.cat([kw["token_type_ids"], torch.zeros_like(extra_ids)], dim=1)
        with torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=att, **kw)
        return out.logits[0].float().log_softmax(-1)

    # -- interface -------------------------------------------------------
    def option_probs(self, image, prompt, options):
        """Each option is scored over its common spellings ("yes", "Yes", "YES")
        so a model that answers in capitals is not penalized."""
        inp = self._inputs(image, prompt)
        forms = {o: list(dict.fromkeys([o, o.capitalize(), o.upper()])) for o in options}
        first = {o: [self._answer_ids(f)[0] for f in fs] for o, fs in forms.items()}
        single = all(len(t) == 1 for ts in first.values() for t in ts)
        tok_sets = [{int(t[0]) for t in first[o]} for o in options] if single else []
        disjoint = single and sum(len(t) for t in tok_sets) == len(set().union(*tok_sets))
        if disjoint:
            # single-token options: one forward pass, read the next-token distribution
            with self.torch.no_grad():
                logits = self.model(**inp).logits[0, -1].float().log_softmax(-1)
            lp = np.array([np.logaddexp.reduce([float(logits[i]) for i in sorted({int(t[0]) for t in first[o]})])
                           for o in options])
        else:
            lp = np.array([np.logaddexp.reduce([self.answer_logprob(image, prompt, f)[0] for f in forms[o]])
                           for o in options])
        p = np.exp(lp - lp.max())
        p = p / p.sum()
        return {o: float(x) for o, x in zip(options, p)}

    def generate(self, image, prompt, n=1, temperature=0.0, max_new_tokens=24):
        torch = self.torch
        inp = self._inputs(image, prompt)
        L = inp["input_ids"].shape[1]
        kw = dict(max_new_tokens=max_new_tokens, output_scores=True, return_dict_in_generate=True)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature, num_return_sequences=n, top_p=1.0, top_k=0)
        else:
            kw.update(do_sample=False, num_return_sequences=1)
        with torch.no_grad():
            out = self.model.generate(**inp, **kw)
        seqs = out.sequences[:, L:]
        scores = torch.stack(out.scores, dim=1).float().log_softmax(-1)  # (n, T, V)
        res = []
        g = getattr(self.model.generation_config, "eos_token_id", None)
        g = g if isinstance(g, list) else [g]
        eos = {t for t in [self.tok.eos_token_id, *g] if t is not None}
        for i in range(seqs.shape[0]):
            lps = []
            for t, tid in enumerate(seqs[i].tolist()):
                if tid in eos or tid == self.tok.pad_token_id:
                    break
                lps.append(float(scores[i, t, tid]))
            text = self.tok.decode(seqs[i], skip_special_tokens=True).strip()
            res.append((text, float(np.mean(lps)) if lps else float("-inf")))
        if not (temperature and temperature > 0) and n > 1:
            res = res * n
        return res

    def answer_logprob(self, image, prompt, answer):
        inp = self._inputs(image, prompt)
        a = self._answer_ids(answer)
        lp = self._forward_with(inp, a)
        L = inp["input_ids"].shape[1]
        tot = 0.0
        for j in range(a.shape[1]):
            tot += float(lp[L - 1 + j, int(a[0, j])])
        return tot, int(a.shape[1])


# ---------------------------------------------------------------------------
class MockAdapter(VLMAdapter):
    """Deterministic fake model for tests. Its confidence depends on how much of
    the image differs from a stored 'evidence fingerprint', so hiding or swapping
    the image changes its confidence in a predictable way. `blind=True` makes a
    model that ignores the image entirely (the case GroundCheck is built to flag)."""

    name = "mock"

    def __init__(self, blind: bool = False, seed: int = 0):
        self.blind = blind
        self.seed = seed
        self.reference: dict[str, np.ndarray] = {}

    def begin_item(self, item_id: str):
        """Called by the runner before each question, so the fake model's stored
        evidence fingerprint belongs to this question's original image."""
        self.reference = {}

    def _key(self, prompt):
        return hashlib.sha256(prompt.splitlines()[0].encode()).hexdigest()[:12]

    def _evidence(self, image, prompt):
        if self.blind:
            return 1.0
        arr = np.asarray(image.convert("L").resize((32, 32)), dtype=float) / 255.0
        k = self._key(prompt)
        ref = self.reference.setdefault(k, arr)
        return float(max(0.0, 1.0 - 4.0 * np.abs(arr - ref).mean()))

    def option_probs(self, image, prompt, options):
        e = self._evidence(image, prompt)
        h = int(self._key(prompt), 16)
        fav = options[h % len(options)]
        peak = 0.5 + 0.45 * e if not self.blind else 0.9
        rest = (1 - peak) / max(1, len(options) - 1)
        if len(options) == 1:
            return {fav: 1.0}
        return {o: (peak if o == fav else rest) for o in options}

    def generate(self, image, prompt, n=1, temperature=0.0, max_new_tokens=24):
        e = self._evidence(image, prompt)
        h = int(self._key(prompt), 16)
        base = ["liver", "lung", "brain", "kidney"][h % 4]
        lp = math.log(0.05 + 0.9 * e)
        if temperature and temperature > 0:
            rng = np.random.default_rng(h + self.seed)
            alt = ["liver", "lung", "brain", "kidney"]
            return [(base if rng.random() < 0.05 + 0.9 * e else alt[rng.integers(4)], lp) for _ in range(n)]
        return [(base, lp)] * n

    def answer_logprob(self, image, prompt, answer):
        e = self._evidence(image, prompt)
        return math.log(0.05 + 0.9 * e), 1


def load_adapter(cfg: dict) -> VLMAdapter:
    kind = cfg.get("adapter", "hf")
    if kind == "mock":
        return MockAdapter(blind=cfg.get("blind", False))
    return HFAdapter(cfg["model_id"], revision=cfg.get("revision"), dtype=cfg.get("dtype", "auto"),
                     device=cfg.get("device", "auto"), max_image_side=cfg.get("max_image_side"),
                     processor_kwargs=cfg.get("processor_kwargs"), trust_remote_code=cfg.get("trust_remote_code", False))
