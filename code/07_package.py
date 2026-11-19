"""07_package.py: build the two release archives and their Zenodo descriptions.

Plain-language summary: your plan has two separate citable releases, each with
its own DOI. This script builds both:
  dist/groundcheck-perturbation-suite-v<version>.zip   (dataset: the suite, codebook, source log)
  dist/groundcheck-method-v<version>.zip               (software: code, configs, results)
plus dist/zenodo_suite.json and dist/zenodo_method.json, the title, authors,
description and license each Zenodo record will show. Author details come from
AUTHORS.json only.

Run:  python code/07_package.py --version 0.1.0
"""
import argparse
import json
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "https://github.com/Racta-1/evidence-grounded-confidence"

SUITE_FILES = ["data/suite/items.csv", "data/suite/variants.csv", "data/suite/qa_report.txt",
               "data/raw/PROVENANCE.txt", "docs/CODEBOOK.md", "docs/LIMITATIONS.md", "LICENSE-DATA.md",
               "code/01_fetch.py", "code/01b_fetch_mimic.py", "code/02_build_suite.py", "code/03_preview_suite.py", "code/provenance.py",
               "src/groundcheck/__init__.py", "src/groundcheck/perturb.py", "src/groundcheck/suite.py", "src/groundcheck/mimic.py",
               "paper/figures/fig1_suite_examples.png", "paper/figures/text_removal_example.txt",
               "requirements.txt", "AUTHORS.json", "CITATION.cff"]
SKIP_DIRS = {"data/raw", "data/interim", "data/suite_private", "dist", ".git", "__pycache__", ".pytest_cache"}
PRIVATE_FILES = ("raw_private.jsonl", "item_scores_private.csv")  # MIMIC: never packaged


def creators():
    a = json.load(open(os.path.join(ROOT, "AUTHORS.json")))
    out = []
    for p in a["authors"] + a.get("collaborators", []):
        c = {"name": f"{p['family']}, {p['given']}", "affiliation": p["affiliation"]}
        if p.get("orcid"):
            c["orcid"] = p["orcid"]
        out.append(c)
    return out


def zip_files(path, files, prefix):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(ROOT, f), os.path.join(prefix, f))


def all_repo_files():
    files = []
    for dp, dns, fns in os.walk(ROOT):
        rel = os.path.relpath(dp, ROOT)
        dns[:] = [d for d in dns if os.path.normpath(os.path.join(rel, d)) not in SKIP_DIRS and d not in SKIP_DIRS]
        for fn in fns:
            if fn in PRIVATE_FILES:
                continue
            files.append(os.path.normpath(os.path.join(rel, fn)))
    files.append("data/raw/PROVENANCE.txt")
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    a = ap.parse_args()
    v = a.version
    dist = os.path.join(ROOT, "dist")
    os.makedirs(dist, exist_ok=True)
    masks = sorted("data/suite/masks/" + m for m in os.listdir(os.path.join(ROOT, "data/suite/masks")))
    sp = os.path.join(dist, f"groundcheck-perturbation-suite-v{v}.zip")
    zip_files(sp, SUITE_FILES + masks, f"groundcheck-perturbation-suite-v{v}")
    mp = os.path.join(dist, f"groundcheck-method-v{v}.zip")
    zip_files(mp, all_repo_files(), f"groundcheck-method-v{v}")
    stats = json.load(open(os.path.join(ROOT, "paper", "stats.json")))["suite"]
    by = stats["items_by_dataset"]
    vo = stats["variants_ok"]
    suite_meta = {"metadata": {
        "upload_type": "dataset",
        "title": f"Evidence Perturbation Suite v{v}: masked-region, swapped-image, and removed-text variants for medical visual question answering",
        "creators": creators(),
        "description": (
            "<p>A test suite for checking whether a vision-language model's confidence in an answer depends on "
            "the evidence behind it. For each of "
            f"{stats['n_items']:,} questions from four public medical visual question answering datasets "
            f"(SLAKE {by.get('SLAKE', 0):,}; VQA-RAD {by.get('VQA-RAD', 0):,}; PathVQA {by.get('PathVQA', 0):,}; "
            f"MedXpertQA multimodal {by.get('MedXpertQA-MM', 0):,}), "
            "the suite defines up to four evidence perturbations: the image region the question depends on is hidden "
            f"({vo.get('SLAKE|mask_region', 0):,} items), an equal-size control patch is hidden elsewhere "
            f"({vo.get('SLAKE|mask_control', 0):,}), the image is swapped for an unrelated one "
            f"({sum(n for k, n in vo.items() if k.endswith('|swap_image')):,}), or the clinical history is removed "
            f"from the question ({vo.get('MedXpertQA-MM|remove_text', 0):,}).</p>"
            "<p>The release contains the item and variant tables, region and control masks, the codebook, the "
            "quality report, the provenance log (source URLs, pinned versions, SHA-256 fingerprints, access dates, "
            "licenses), and the scripts that rebuild every perturbed image from the original sources. "
            "Source images are not redistributed; they are downloaded from their original hosts. The release also "
            "includes the script that builds MIMIC-CXR items locally for users with credentialed PhysioNet access; "
            "no MIMIC data is included.</p>"),
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": v,
        "keywords": ["medical visual question answering", "uncertainty estimation", "evidence perturbation",
                     "vision-language models", "calibration", "benchmark", "SLAKE", "VQA-RAD", "PathVQA", "MedXpertQA", "MIMIC-CXR"],
        "related_identifiers": [{"identifier": REPO, "relation": "isSupplementedBy", "resource_type": "software"}],
    }}
    method_meta = {"metadata": {
        "upload_type": "software",
        "title": f"GroundCheck: Evidence-Grounded Confidence for clinical vision-language models, code and configurations v{v}",
        "creators": creators(),
        "description": (
            "<p>Code and configurations for an evidence-grounded confidence method. The method measures whether a "
            "vision-language model's confidence in its answer survives when the image evidence behind the answer is "
            "masked, swapped, or removed, flags answers whose confidence does not depend on their evidence, and "
            "lowers the reported confidence accordingly. Baselines included: token probability, verbalized "
            "confidence, sampling consistency, split-conformal prediction sets, and Platt scaling. Configurations "
            "cover three matched pairs of general and medical open-weight models. Includes tests, the scoring "
            "pipeline, and the results available at release.</p>"),
        "access_right": "open",
        "license": "mit",
        "version": v,
        "keywords": ["uncertainty estimation", "calibration", "vision-language models", "clinical AI",
                     "reliability", "conformal prediction", "evidence grounding"],
        "related_identifiers": [{"identifier": REPO, "relation": "isSupplementTo", "resource_type": "software"}],
    }}
    json.dump(suite_meta, open(os.path.join(dist, "zenodo_suite.json"), "w"), indent=2)
    json.dump(method_meta, open(os.path.join(dist, "zenodo_method.json"), "w"), indent=2)
    for p in (sp, mp):
        print(f"{p}  {os.path.getsize(p) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
