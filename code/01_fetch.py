"""01_fetch.py: download the source datasets and log where each file came from.

Plain-language summary: this script downloads four free, public medical image
question sets (SLAKE, VQA-RAD, PathVQA, MedXpertQA) from Hugging Face, pinned to an exact version,
and writes one line per file to data/raw/PROVENANCE.txt (date, size, fingerprint,
source link, license). You do not need to open this file; run it once.

MIMIC-CXR is credentialed and is downloaded separately by code/01b_fetch_mimic.py.

Run:  python code/01_fetch.py
"""
import os
import subprocess
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
LOG = os.path.join(RAW, "PROVENANCE.txt")
PROV = os.path.join(ROOT, "code", "provenance.py")

# Pinned to exact repository commits so the download is the same every time.
SOURCES = [
    {
        "name": "slake",
        "repo": "BoKelvin/SLAKE",
        "rev": "a9083ce6c34ac3ffb17671a605962924d8a8f9e9",
        "license": "CC BY 4.0 (per Hugging Face dataset card; original: Liu et al., ISBI 2021, https://www.med-vqa.com/slake/)",
        "files": ["train.json", "validation.json", "test.json", "mask.txt", "imgs.zip"],
    },
    {
        "name": "vqa_rad",
        "repo": "flaviagiammarino/vqa-rad",
        "rev": "bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9",
        "license": "CC0 1.0 (per Hugging Face dataset card; original: Lau et al., Scientific Data 2018, https://osf.io/89kps/)",
        "files": ["data/test-00000-of-00001-e5bc3d208bb4deeb.parquet",
                  "data/train-00000-of-00001-eb8844602202be60.parquet"],
    },
    {
        "name": "pathvqa",
        "repo": "flaviagiammarino/path-vqa",
        "rev": "1685832883334b5bb5beaf4e4b333fdeecaa4ad9",
        "license": "MIT (per Hugging Face dataset card; original: He et al., arXiv:2003.10286, https://github.com/UCSD-AI4H/PathVQA)",
        "files": ["data/test-00000-of-00003-e9adadb4799f44d3.parquet",
                  "data/test-00001-of-00003-7ea98873fc919813.parquet",
                  "data/test-00002-of-00003-1628308435019820.parquet"],
    },
    {
        "name": "medxpertqa",
        "repo": "TsinghuaC3I/MedXpertQA",
        "rev": "7e7c465a68eb2b866926bfa59c8c9d17a8daba65",
        "license": "MIT (per Hugging Face dataset card; original: Zuo et al., arXiv:2501.18362, https://github.com/TsinghuaC3I/MedXpertQA)",
        "files": ["MM/test.jsonl", "MM/dev.jsonl", "images.zip"],
    },
]


def logged_fingerprints():
    """name -> sha256 already recorded in PROVENANCE.txt (the committed record)."""
    out = {}
    if os.path.exists(LOG):
        for line in open(LOG):
            parts = [x.strip() for x in line.split("|")]
            if len(parts) > 3 and parts[3].startswith("sha256:"):
                out.setdefault(parts[1], set()).add(parts[3][7:])
    return out


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(src):
    known = logged_fingerprints()
    outdir = os.path.join(RAW, src["name"])
    os.makedirs(outdir, exist_ok=True)
    for f in src["files"]:
        url = f"https://huggingface.co/datasets/{src['repo']}/resolve/{src['rev']}/{f}"
        dest = os.path.join(outdir, os.path.basename(f))
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print("exists, skipping:", dest)
            continue
        print("downloading", url)
        urllib.request.urlretrieve(url, dest)
        name = os.path.basename(dest)
        if name in known:
            # already in the committed record: check the download is byte-identical instead of re-logging
            if sha256(dest) in known[name]:
                print(f"  fingerprint matches the logged record: {name}")
            else:
                sys.exit(f"FINGERPRINT MISMATCH for {name}: the source changed since it was logged. "
                         "Stop and tell Claude; results would not be comparable.")
            continue
        subprocess.run([sys.executable, PROV, dest, url, "--log", LOG,
                        "--note", f"license: {src['license']}; pinned commit {src['rev']}"], check=True)
    unzip = {"slake": ("imgs.zip", "imgs"), "medxpertqa": ("images.zip", "images")}
    if src["name"] in unzip:
        zname, dname = unzip[src["name"]]
        if not os.path.isdir(os.path.join(outdir, dname)):
            with zipfile.ZipFile(os.path.join(outdir, zname)) as zf:
                members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
                zf.extractall(outdir, members=members)


if __name__ == "__main__":
    for s in SOURCES:
        fetch(s)
    print("done; see", LOG)
