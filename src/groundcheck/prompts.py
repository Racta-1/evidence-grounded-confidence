"""Prompt wording and answer matching.

Plain-language summary: the exact instructions sent to each model, and the rules
for deciding whether a model's answer matches the correct answer. Keeping these
in one place means every model is asked the same way.
"""
from __future__ import annotations

import re

INSTRUCT = {
    "closed": "Answer with yes or no.",
    "mcq": "Answer with the letter of the correct option only.",
    "open": "Answer with a single word or short phrase.",
}
VERBAL_SUFFIX = ("\nAfter your answer, write a new line in the form 'Confidence: N', where N is a number "
                 "from 0 to 100 giving how sure you are that your answer is correct.")


def build_prompt(question: str, answer_type: str, verbalized: bool = False) -> str:
    p = f"{question.strip()}\n{INSTRUCT[answer_type]}"
    return p + (VERBAL_SUFFIX if verbalized else "")


def option_strings(answer_type: str, options: list[str]) -> list[str]:
    """Surface strings the model is scored on for closed-set items."""
    if answer_type == "closed":
        return ["yes", "no"]
    return list(options)


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"confidence\s*:.*", "", s, flags=re.S)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_choice(text: str, answer_type: str, options: list[str]) -> str:
    """Map free text to one option (closed/mcq), or the normalized text (open)."""
    t = text.strip()
    if answer_type == "mcq":
        m = re.search(r"\b([A-J])\b", t.upper().replace("(", " ").replace(")", " "))
        return m.group(1) if m and m.group(1) in options else norm(t)[:1].upper()
    if answer_type == "closed":
        n = norm(t)
        if n.startswith("yes"):
            return "yes"
        if n.startswith("no"):
            return "no"
        return n.split(" ")[0] if n else ""
    first_line = t.splitlines()[0] if t else ""
    return norm(first_line)


def is_correct(pred: str, gold: str, answer_type: str) -> bool:
    if answer_type in ("closed", "mcq"):
        return norm(pred) == norm(gold)
    p, g = norm(pred), norm(gold)
    if not p:
        return False
    return p == g or (len(g) > 0 and re.search(rf"\b{re.escape(g)}\b", p) is not None)


def parse_verbal_conf(text: str) -> float | None:
    m = re.search(r"confidence\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*%?", text, flags=re.I)
    if not m:
        return None
    v = float(m.group(1))
    return min(100.0, v) / 100.0
