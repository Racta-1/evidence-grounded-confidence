"""01b_fetch_mimic.py: download the MIMIC-CXR questions and only the X-ray images they use.

Plain-language summary: MIMIC-CXR is not public. It needs your own credentialed
PhysioNet account and a signed data use agreement. Run this on YOUR machine (the
cluster), never in a shared or public place. It:
  1. downloads the question files of MIMIC-Ext-MIMIC-CXR-VQA (v1.0.0) with your login;
  2. finds the test-split questions;
  3. downloads only the chest X-ray images those questions use, from MIMIC-CXR-JPG
     (v2.1.0), instead of the whole 500+ GB collection.
Everything lands in data/raw/mimic/, which is excluded from GitHub. Your password
is asked for on screen and never saved.

Before running, sign in at https://physionet.org and accept the data use agreement on
both project pages:
  https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/
  https://physionet.org/content/mimic-cxr-jpg/2.1.0/

Run:  python code/01b_fetch_mimic.py --user YOUR_PHYSIONET_USERNAME
      (optional) --max-images 600   cap the number of X-rays downloaded (default: all used by the test split)
"""
import argparse
import base64
import getpass
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from groundcheck import mimic as MM  # noqa: E402

OUT = os.path.join(ROOT, "data", "raw", "mimic")
VQA_URL = "https://physionet.org/files/mimic-ext-mimic-cxr-vqa/1.0.0/"
JPG_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
PROV = os.path.join(ROOT, "code", "provenance.py")
LOG = os.path.join(OUT, "PROVENANCE_MIMIC.txt")  # stays private: file names carry MIMIC IDs


def fetch_vqa(user, password):
    """Mirror the question-set project with wget (handles PhysioNet's folder listing)."""
    dest = os.path.join(OUT, "vqa")
    os.makedirs(dest, exist_ok=True)
    if not shutil.which("wget"):
        sys.exit("wget is not installed. Ask your cluster help desk, or download the folder by hand "
                 f"from {VQA_URL} into {dest}.")
    # credentials go in a private temporary wgetrc (mode 600), never on the command line,
    # where other users of a shared cluster could see them in the process list
    import tempfile
    fd, rc = tempfile.mkstemp(prefix="wgetrc_")
    os.chmod(rc, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"user = {user}\npassword = {password}\n")
    cmd = ["wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3", "-P", dest, "-R", "index.html*", VQA_URL]
    print("downloading the question files (this can take several minutes)...")
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                           env={**os.environ, "WGETRC": rc})
    finally:
        os.remove(rc)
    if r.returncode not in (0, 8):  # 8 = some server responses were errors (often harmless 404s on listings)
        sys.exit("wget failed. Check your username and password, and that you accepted the data use "
                 "agreement on the project page.\n" + r.stderr[-1500:])
    return dest


def download(url, dest, auth_header, tries=3):
    req = urllib.request.Request(url, headers={"Authorization": auth_header})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest + ".part", "wb") as f:
                shutil.copyfileobj(resp, f)
            os.replace(dest + ".part", dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                sys.exit(f"PhysioNet refused access ({e.code}) for {url}. Did you accept the MIMIC-CXR-JPG "
                         "data use agreement at https://physionet.org/content/mimic-cxr-jpg/2.1.0/ ?")
            if e.code == 404:
                return False
        except Exception:
            pass
        time.sleep(2 * (k + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--skip-vqa-download", action="store_true",
                    help="use question files already in data/raw/mimic/vqa")
    a = ap.parse_args()
    password = os.environ.get("PHYSIONET_PASSWORD") or getpass.getpass("PhysioNet password: ")
    os.makedirs(OUT, exist_ok=True)
    vqa_dir = os.path.join(OUT, "vqa") if a.skip_vqa_download else fetch_vqa(a.user, password)
    test_json = MM.find_test_json(vqa_dir)
    if not test_json:
        found = [os.path.relpath(p, vqa_dir) for p in glob.glob(os.path.join(vqa_dir, "**", "*.json"), recursive=True)]
        sys.exit(f"Could not find the test question file. JSON files found: {found[:20]}. "
                 "Send this message to Claude.")
    subprocess.run([sys.executable, PROV, test_json, VQA_URL, "--log", LOG,
                    "--note", "MIMIC-Ext-MIMIC-CXR-VQA v1.0.0 test split; PhysioNet Credentialed Health Data License 1.5.0"],
                   check=True)
    entries = MM.load_entries(test_json)
    paths = MM.image_paths(entries)
    if a.max_images:
        paths = paths[: a.max_images]
    print(f"{len(entries)} test questions use {len(MM.image_paths(entries))} images; downloading {len(paths)}")
    auth = "Basic " + base64.b64encode(f"{a.user}:{password}".encode()).decode()
    img_root = os.path.join(OUT, "mimic-cxr-jpg")
    ok = missing = 0
    for n, rel in enumerate(paths, 1):
        dest = os.path.join(img_root, rel)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            ok += 1
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if download(JPG_URL + rel, dest, auth):
            ok += 1
        else:
            missing += 1
        if n % 50 == 0:
            print(f"  {n}/{len(paths)} images")
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d')} | mimic-cxr-jpg images | {ok} downloaded or present, {missing} missing | "
                f"{JPG_URL} | PhysioNet Credentialed Health Data License 1.5.0\n")
    print(f"done: {ok} images ready, {missing} missing. Next: python code/02_build_suite.py")


if __name__ == "__main__":
    main()
