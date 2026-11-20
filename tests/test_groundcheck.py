"""Automatic tests. Run:  python -m pytest -q tests

Plain-language summary: small checks that each piece does what it claims, plus
one end-to-end run with a fake model. The key test: a fake model that ignores
the image entirely must be flagged by the method, and a fake model that uses
the image must not be.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from groundcheck import method as M  # noqa: E402
from groundcheck import metrics as MT  # noqa: E402
from groundcheck import perturb as P  # noqa: E402
from groundcheck import prompts as PR  # noqa: E402


# --- region targeting --------------------------------------------------------
PRESENT = ["Left Lung", "Right Lung", "Liver", "Liver Cancer", "Heart", "Spleen"]


def test_side_specific_beats_generic():
    labs, rule = P.target_labels("What color does the left lung show?", "Black", "Color", PRESENT)
    assert labs == ["Left Lung"] and rule == "R1_named"


def test_generic_matches_both_sides():
    labs, _ = P.target_labels("How many lungs are in this image?", "2", "Quantity", PRESENT)
    assert labs == ["Left Lung", "Right Lung"]


def test_organ_includes_its_lesion():
    labs, _ = P.target_labels("Does the picture contain liver?", "Yes", "Organ", PRESENT)
    assert set(labs) == {"Liver", "Liver Cancer"}


def test_abnormality_rule():
    labs, rule = P.target_labels("Are there abnormalities in this image?", "Yes", "Abnormality", PRESENT)
    assert labs == ["Liver Cancer"] and rule == "R2_abnormality"


def test_no_region_for_global_question():
    labs, rule = P.target_labels("What modality is used to take this image?", "CT", "Modality", PRESENT)
    assert labs == [] and rule == "R0_none"


def test_label_mask_detection():
    legend = {70: "Liver", 175: "Liver Cancer"}
    assert P.is_label_mask(np.array([[0, 70], [175, 0]]), legend)
    assert not P.is_label_mask(np.arange(16).reshape(4, 4), legend)


# --- control mask ---------------------------------------------------------------
def test_control_same_area_no_overlap():
    gray = np.full((100, 100), 128, np.uint8)
    region = np.zeros((100, 100), bool)
    region[10:30, 10:30] = True
    ctrl = P.control_region(region, gray, "x")
    assert ctrl is not None and ctrl.sum() == region.sum() and not (ctrl & region).any()
    assert np.array_equal(ctrl, P.control_region(region, gray, "x"))  # deterministic


# --- text removal ---------------------------------------------------------------
def test_remove_context_keeps_final_question():
    q = ("A 29-year-old woman has a painless mass. It appeared 4 months after delivery. "
         "What is the most likely diagnosis?\nAnswer Choices: (A) Cyst (B) Tumor")
    new, removed, rule = P.remove_context(q)
    assert new.startswith("What is the most likely diagnosis?") and "(A) Cyst" in new
    assert removed > 0 and rule == "T1_final_question_only"
    assert P.remove_context("What is shown?\nAnswer Choices: (A) x (B) y")[0] is None


# --- prompts and matching -------------------------------------------------------
def test_answer_matching():
    assert PR.parse_choice("(C) Radial head", "mcq", list("ABCDE")) == "C"
    assert PR.parse_choice("Yes, it is.", "closed", ["yes", "no"]) == "yes"
    assert PR.is_correct("the liver", "Liver", "open")
    assert not PR.is_correct("lung", "Liver", "open")
    assert PR.is_correct("Lungs.", "Lung", "open")              # plural alone is not an error
    assert PR.is_correct("the right lung", "Right Lung", "open")
    assert not PR.is_correct("lung", "Right Lung, Lower Left", "open")
    assert PR.is_correct("pancreas", "Pancreas", "open")
    assert PR.parse_verbal_conf("B\nConfidence: 85") == 0.85


# --- metrics --------------------------------------------------------------------
def test_metrics_known_values():
    rng = np.random.default_rng(0)
    c = rng.uniform(0, 1, 20000)
    y = rng.uniform(0, 1, 20000) < c          # perfectly calibrated by construction
    assert MT.ece(c, y) < 0.02
    assert MT.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert MT.brier([1.0, 0.0], [1, 0]) == 0.0


# --- method ---------------------------------------------------------------------
def test_survival_is_chance_normalized():
    row = pd.Series({"chance": 0.5, "p_a0__original": 0.9, "p_a0__swap_image": 0.5})
    assert M.survival_per_variant(row)["swap_image"] == 0.0
    row = pd.Series({"chance": 0.5, "p_a0__original": 0.9, "p_a0__swap_image": 0.9})
    assert M.survival_per_variant(row)["swap_image"] == 1.0


def test_control_used_as_reference():
    row = pd.Series({"chance": 0.0, "p_a0__original": 0.9, "p_a0__mask_control": 0.6, "p_a0__mask_region": 0.3})
    assert abs(M.survival_per_variant(row)["mask_region"] - 0.5) < 1e-9


def test_groundcheck_lowers_only_surviving_confidence():
    out = M.groundcheck_adjust([0.9, 0.9], [1.0, 0.0], lam=0.5)
    assert out[0] == pytest.approx(0.45) and out[1] == pytest.approx(0.9)


def test_conformal_coverage():
    rng = np.random.default_rng(1)
    p_true = rng.uniform(0.3, 1, 1000)
    q = M.conformal_qhat(p_true, alpha=0.1)
    assert np.mean(1 - p_true <= q) >= 0.9


# --- end to end with fake models on the real suite ---------------------------------
HAVE_SUITE = os.path.exists(os.path.join(ROOT, "data", "suite", "items.csv"))


@pytest.mark.skipif(not HAVE_SUITE, reason="suite not built; run code/01 and code/02 first")
@pytest.mark.parametrize("blind", [False, True])
def test_end_to_end_blind_model_is_flagged(tmp_path, blind):
    from groundcheck.models import MockAdapter
    from groundcheck.run import run_items, select_items
    from groundcheck.score import item_frame, score
    from groundcheck.suite import Suite
    s = Suite(ROOT)
    ids = select_items(s, ["SLAKE"], ["cal", "test"], 30, ["mask_region"])
    out = tmp_path / "raw.jsonl"
    run_items(MockAdapter(blind=blind), s, ids, str(out), {"consistency_k": 3, "verbalized": False}, log=lambda *_: None)
    df = item_frame(str(out), s)
    assert len(df) == 30
    df, res = score(df)
    u = df["u__swap_image"].mean()
    if blind:
        assert u > 0.95 and df.flag.mean() > 0.9
    else:
        # the fake sighted model compares whole pictures, so it is tested on swaps only
        assert u < 0.4 and (df.conf_token.ge(0.8) & df["u__swap_image"].ge(0.8)).mean() <= 0.15


# --- MIMIC reader (synthetic entries; real data never enters the repository) ---------
def test_mimic_reader(tmp_path):
    from groundcheck import mimic as MM
    d = tmp_path / "vqa" / "dataset"
    d.mkdir(parents=True)
    entries = [
        {"idx": 1, "question": "Is there effusion?", "answer": ["yes"], "subject_id": 10000032,
         "image_path": "p10/p10000032/s50414267/abc.jpg"},
        {"idx": 2, "question": "List all findings.", "answer": ["a", "b"],
         "image_path": "files/p11/p11000001/s1/def.jpg"},
        {"idx": 3, "question": "Gender?", "answer": "female", "subject_id": 11, "study_id": 22, "image_id": "ghi"},
    ]
    (d / "test.json").write_text(json.dumps(entries))
    tj = MM.find_test_json(str(tmp_path / "vqa"))
    assert tj.endswith("test.json")
    es = MM.load_entries(tj)
    assert MM.image_rel_path(es[0]) == "files/p10/p10000032/s50414267/abc.jpg"
    assert MM.image_rel_path(es[1]) == "files/p11/p11000001/s1/def.jpg"
    assert MM.image_rel_path(es[2]) == "files/p11/p11/s22/ghi.jpg"
    assert MM.normalize_answer(es[0]["answer"]) == "yes"
    assert MM.normalize_answer(es[1]["answer"]) is None      # multi-answer questions excluded (D11)
    assert MM.normalize_answer(es[2]["answer"]) == "female"
    assert MM.patient_id(es[1], MM.image_rel_path(es[1])) == "p11000001"


def test_private_items_are_routed_to_private_file():
    from groundcheck.run import private_path
    from groundcheck.suite import is_private
    assert is_private("mimic-123") and not is_private("slake-1")
    assert private_path("results/x/raw.jsonl") == "results/x/raw_private.jsonl"


def test_choice_questions():
    assert PR.choice_options("Which organ is abnormal, heart or lung?", "Lung") == ["heart", "lung"]
    assert PR.choice_options("Which is the biggest in this image,lung,liver or heart?", "Lung") == ["lung", "liver", "heart"]
    assert PR.choice_options("is this supratentorial or infratentorial?", "supratentorial") == ["supratentorial", "infratentorial"]
    assert PR.choice_options("are there multiple or just 1 metastatic focus?", "one") is None   # answer not an option
    assert PR.choice_options("What modality is this?", "CT") is None
    opts = ["heart", "lung"]
    assert PR.parse_choice("The lungs.", "choice", opts) == "lung"
    assert PR.is_correct("lung", "Lung", "choice") and not PR.is_correct("heart", "Lung", "choice")
    assert "Answer with one of: heart, lung." in PR.build_prompt("Which organ is abnormal, heart or lung?", "choice", options=opts)
