# Running the six models on a GPU

*Plain-language header: the real results need a graphics card (GPU), which Claude's workspace does not have. This guide runs everything on a GPU you can reach, with copy-paste steps. Your time: about 1 hour, spread out. Machine time: roughly 6 to 12 hours per model with all five datasets (PathVQA is the largest), so about 2 to 3 days for all six if run one after another. That figure is an estimate; the first model's run tells you the real number.*

**What you need:** a GPU with at least 24 GB of memory (an NVIDIA L4, A10G, A100 or better). Gemma and MedGemma also need you to accept their terms on Hugging Face first: open https://huggingface.co/google/gemma-3-4b-it and https://huggingface.co/google/medgemma-4b-it, sign in, and click the button to agree (2 min).

## Option A: Google Colab (easiest)

1. Go to https://colab.research.google.com and choose **File > New notebook**.
2. Choose **Runtime > Change runtime type**, pick **L4 GPU** or **A100 GPU**, then click **Save**. (These need Colab Pro, about $10 a month.)
3. Give Colab a key to your private repository (one time, 3 min). Create a read-only key at https://github.com/settings/personal-access-tokens/new: name it "colab read", choose **Only select repositories** and pick `evidence-grounded-confidence`, under **Repository permissions** set **Contents** to **Read-only**, then click **Generate token** and copy it. In Colab, click the key icon on the left (**Secrets**), click **Add new secret**, name it `GH_READ`, paste the key as the value, and switch on **Notebook access**. The key stays in Colab's secrets, never in the project.
4. Paste each block below into a cell and press the play button.

```python
from google.colab import userdata
gh_read = userdata.get("GH_READ")
!git clone https://{gh_read}@github.com/Racta-1/evidence-grounded-confidence.git
%cd evidence-grounded-confidence
!pip install -q -r requirements.txt
!python code/01_fetch.py
!python code/02_build_suite.py
```
You should see **9 lines starting with PASS** at the bottom. If any says FAIL, stop and send Claude the output.

5. Sign in to Hugging Face for the gated models. Create a **read** token at https://huggingface.co/settings/tokens. Then run the cell below and paste the token when asked. The token is not saved in the project.
```python
from huggingface_hub import login; login()
```

6. Run one model at a time. Each run saves as it goes, so if Colab disconnects, run the same line again and it picks up where it stopped.
```bash
!python code/04_run_model.py configs/qwen2_5_vl_7b.yaml
!python code/04_run_model.py configs/lingshu_7b.yaml
!python code/04_run_model.py configs/gemma3_4b.yaml
!python code/04_run_model.py configs/medgemma_4b.yaml
!python code/04_run_model.py configs/llava16_mistral_7b.yaml
!python code/04_run_model.py configs/llavamed15_mistral_7b.yaml
```

7. Score and draw the figures:
```bash
!python code/05_score.py
!python code/06_figures.py
```

8. Save the results. Run the cell below, then download `results_bundle.zip` from the Files panel on the left (folder icon).
```bash
!zip -r results_bundle.zip results paper
```
Send that zip to Claude in a new message. It is everything needed for the verification step.

**Tip:** before the full run, test with 20 questions per dataset to catch problems early:
`!python code/04_run_model.py configs/qwen2_5_vl_7b.yaml --limit 20`. Then delete the `results/qwen2_5_vl_7b` folder before the full run, so the test answers are not mixed in.

## MIMIC-CXR (on your own machine only)

MIMIC-CXR needs your credentialed PhysioNet account. Do this step only on a machine you control (your cluster account), never on a shared notebook you might share with others.

1. Sign in at https://physionet.org. On each of these pages, scroll to **Files** and sign the data use agreement if asked: https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/ and https://physionet.org/content/mimic-cxr-jpg/2.1.0/ (5 min).
2. In the project folder run:
   ```
   python code/01b_fetch_mimic.py --user YOUR_PHYSIONET_USERNAME
   ```
   It asks for your PhysioNet password; what you type is not shown. It downloads the question files, then only the test-split chest X-rays, not the whole 500+ GB collection. It ends with `done: N images ready`. If it stops with a message about a missing test file or refused access, send Claude that message. It never contains your password.
3. Run `python code/02_build_suite.py` again. After the public report you'll see a second report titled **PRIVATE (MIMIC-CXR ...)** with 8 PASS lines.
4. Check `git status`. Nothing under `data/raw/mimic`, `data/suite_private`, or any file ending in `_private.jsonl` or `_private.csv` should appear; the project ignores them on purpose. MIMIC outputs stay on this machine. Only summary numbers go into `paper/stats.json`.

## On a shared multi-GPU machine (such as wukong)

Each run uses exactly **one** GPU. Pick a free one first, so you don't slow down other people's jobs:
```
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```
Choose a GPU with near-zero memory used and 0% utilization (say GPU 4), and put `CUDA_VISIBLE_DEVICES=4` in front of the command:
```
CUDA_VISIBLE_DEVICES=4 python code/04_run_model.py configs/qwen2_5_vl_7b.yaml
```
The machine's driver supports CUDA up to the version `nvidia-smi` shows at the top right. For CUDA 12.x, install PyTorch with `python -m pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu126`.

## Running as a job: `scripts/run_groundcheck.sh`

One script runs any set of models on one GPU, then scores and draws figures. Read its first 20 lines for all the options.
- With Slurm (`which sbatch` prints a path): `sbatch scripts/run_groundcheck.sh qwen2_5_vl_7b --limit 20`. Check it with `squeue -u $USER`; the output goes to `logs/groundcheck_<job number>.out`.
- Without Slurm: `nohup bash scripts/run_groundcheck.sh qwen2_5_vl_7b --limit 20 > logs/test.log 2>&1 &`, then `tail -f logs/test.log`. It picks the least-busy GPU itself.
- Leave out the model names to run all six, one after another.

## Option B: The University of Alabama research computing cluster

Log in to the cluster and run `git clone https://github.com/Racta-1/evidence-grounded-confidence.git` (when asked for a password, paste the read-only key from step 3 of Option A), then `cd evidence-grounded-confidence && pip install --user -r requirements.txt && python code/01_fetch.py && python code/02_build_suite.py`.

To get later updates on either machine: `git pull` inside the project folder. If an older version of the project changed `data/raw/PROVENANCE.txt` on your machine and `git pull` refuses, run `git checkout -- data/raw/PROVENANCE.txt` and then `git pull` again.

If you have an account with UA's research computing (Office of Information Technology), the job file below works on most clusters that use the Slurm scheduler. Ask the help desk which **partition** and **module** names to use. They differ by cluster.

```bash
#!/bin/bash
#SBATCH --job-name=groundcheck
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=<GPU partition name from the help desk>
module load <python or cuda module from the help desk>
cd $HOME/evidence-grounded-confidence
for c in qwen2_5_vl_7b lingshu_7b gemma3_4b medgemma_4b llava16_mistral_7b llavamed15_mistral_7b; do
  python code/04_run_model.py configs/$c.yaml
done
python code/05_score.py && python code/06_figures.py
```
Run `code/01b_fetch_mimic.py` once, interactively, before submitting this job, so the MIMIC items exist; the job cannot type your password.
Save it as `run_groundcheck.sh` and submit it with `sbatch run_groundcheck.sh`. Store your Hugging Face token with `huggingface-cli login` on the cluster, never in this file.

## What each run produces

`results/<model>/raw.jsonl` (every answer), `run_meta.json` (software versions, start and finish time), and after scoring `item_scores.csv` and `metrics.json`. A run is complete when `run_meta.json` contains a `finished` line.
