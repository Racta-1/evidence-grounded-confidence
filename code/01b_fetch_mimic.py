"""01b_fetch_mimic.py: download the MIMIC-CXR questions and only the X-ray images they use.

Plain-language summary: MIMIC-CXR is not public. It needs your own credentialed
PhysioNet account and a signed data use agreement. Run this on YOUR machine (the
cluster), never in a shared or public place. It:
  1. downloads the question files of MIMIC-Ext-MIMIC-CXR-VQA (v1.0.0) with your login;
  2. finds the test-split questions;
  3. downloads only the chest X-ray images those questions use, from MIMIC-CXR-JPG
     (v2.1.0), instead of the whole 500+ GB collection. Both steps use wget.
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
import getpass
import glob
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from groundcheck import mimic as MM  # noqa: E402

OUT = os.path.join(ROOT, "data", "raw", "mimic")
VQA_URL = "https://physionet.org/files/mimic-ext-mimic-cxr-vqa/1.0.0/"
JPG_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
PROV = os.path.join(ROOT, "code", "provenance.py")
LOG = os.path.join(OUT, "PROVENANCE_MIMIC.txt")  # stays private: file names carry MIMIC IDs


class PrivateWgetrc:
    """Credentials in a temporary wgetrc readable only by you (mode 600), deleted afterwards.
    Never on the command line, where other cluster users could see them."""

    def __init__(self, user, password):
        self.user, self.password = user, password

    def __enter__(self):
        import tempfile
        fd, self.path = tempfile.mkstemp(prefix="wgetrc_")
        os.chmod(self.path, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"user = {self.user}\npassword = {self.password}\n")
        return {**os.environ, "WGETRC": self.path}

    def __exit__(self, *exc):
        os.remove(self.path)


def need_wget():
    if not shutil.which("wget"):
        sys.exit("wget is not installed on this machine. Ask your cluster help desk to load it "
                 "(often: module load wget), then run this again.")


def fetch_vqa(user, password):
    """Mirror the question-set project with wget (handles PhysioNet's folder listing)."""
    dest = os.path.join(OUT, "vqa")
    os.makedirs(dest, exist_ok=True)
    need_wget()
    cmd = ["wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3", "--auth-no-challenge",
           "-P", dest, "-R", "index.html*", VQA_URL]
    print("downloading the question files (this can take several minutes)...")
    with PrivateWgetrc(user, password) as env:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)
    if r.returncode not in (0, 8):  # 8 = some server responses were errors (often harmless 404s on listings)
        sys.exit("wget failed. Check your username and password, and that you accepted the data use "
                 "agreement on the project page.\n" + r.stderr[-1500:])
    return dest


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
    img_root = os.path.join(OUT, "mimic-cxr-jpg")
    todo = [rel for rel in paths if not (os.path.exists(os.path.join(img_root, rel))
                                         and os.path.getsize(os.path.join(img_root, rel)) > 0)]
    print(f"{len(paths) - len(todo)} already present; downloading {len(todo)} with wget...")
    if todo:
        need_wget()
        url_list = os.path.join(OUT, "image_urls.txt")
        with open(url_list, "w") as f:
            f.write("\n".join(JPG_URL + rel for rel in todo) + "\n")
        # -x keeps the folder layout; -nH --cut-dirs=3 drops "physionet.org/files/mimic-cxr-jpg/2.1.0"
        cmd = ["wget", "-nv", "-c", "-x", "-nH", "--cut-dirs=3", "--auth-no-challenge",
               "-P", img_root, "-i", url_list]
        with PrivateWgetrc(a.user, password) as env:
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)
        if r.returncode == 6:
            sys.exit("PhysioNet rejected the username or password for the image download.")
        if r.returncode == 8 and "403" in (r.stderr or ""):
            sys.exit("PhysioNet refused access (403) to MIMIC-CXR-JPG. Sign in at "
                     "https://physionet.org/content/mimic-cxr-jpg/2.1.0/ and check that the Files section "
                     "shows a file list. If it asks you to sign a data use agreement, sign it and run:\n"
                     f"  python code/01b_fetch_mimic.py --user {a.user} --skip-vqa-download")
    ok = sum(1 for rel in paths if os.path.exists(os.path.join(img_root, rel)) and os.path.getsize(os.path.join(img_root, rel)) > 0)
    missing = len(paths) - ok
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d')} | mimic-cxr-jpg images | {ok} downloaded or present, {missing} missing | "
                f"{JPG_URL} | PhysioNet Credentialed Health Data License 1.5.0\n")
    print(f"done: {ok} images ready, {missing} missing. Next: python code/02_build_suite.py")


if __name__ == "__main__":
    main()
