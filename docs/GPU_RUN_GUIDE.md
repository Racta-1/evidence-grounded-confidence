# Running the six models on a GPU

*Plain-language header: the real results need a graphics card (GPU), which Claude's workspace does not have. This guide runs everything on a GPU you can reach, with copy-paste steps. Your time: about 1 hour, spread out. Machine time: roughly 2 to 4 hours per model, so 12 to 24 hours for all six. That figure is an estimate; the first model's run tells you the real number.*

**What you need:** a GPU with at least 24 GB of memory (an NVIDIA L4, A10G, A100 or better). Gemma and MedGemma also need you to accept their terms on Hugging Face first: open https://huggingface.co/google/gemma-3-4b-it and https://huggingface.co/google/medgemma-4b-it, sign in, and click the button to agree (2 min).

## Option A: Google Colab (easiest)

1. Go to https://colab.research.google.com and choose **File > New notebook**.
2. Choose **Runtime > Change runtime type**, pick **L4 GPU** or **A100 GPU**, then click **Save**. (These need Colab Pro, about $10 a month.)
3. Upload the project. Click the folder icon on the left, then the upload icon (a page with an up arrow), and choose `egc-repo-v0.1.0.zip`. Wait until it appears in the file list (about 1 minute).
4. Paste each block below into a cell and press the play button.

```bash
!unzip -q egc-repo-v0.1.0.zip
%cd egc
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

## Option B: The University of Alabama research computing cluster

Copy `egc-repo-v0.1.0.zip` to the cluster (for example with `scp egc-repo-v0.1.0.zip <your username>@<cluster address>:~`), then log in and run `unzip egc-repo-v0.1.0.zip && cd egc && pip install --user -r requirements.txt && python code/01_fetch.py && python code/02_build_suite.py`.

If you have an account with UA's research computing (Office of Information Technology), the job file below works on most clusters that use the Slurm scheduler. Ask the help desk which **partition** and **module** names to use. They differ by cluster.

```bash
#!/bin/bash
#SBATCH --job-name=egc
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=<GPU partition name from the help desk>
module load <python or cuda module from the help desk>
cd $HOME/egc
for c in qwen2_5_vl_7b lingshu_7b gemma3_4b medgemma_4b llava16_mistral_7b llavamed15_mistral_7b; do
  python code/04_run_model.py configs/$c.yaml
done
python code/05_score.py && python code/06_figures.py
```
Save it as `run_egc.sh` and submit it with `sbatch run_egc.sh`. Store your Hugging Face token with `huggingface-cli login` on the cluster, never in this file.

## What each run produces

`results/<model>/raw.jsonl` (every answer), `run_meta.json` (software versions, start and finish time), and after scoring `item_scores.csv` and `metrics.json`. A run is complete when `run_meta.json` contains a `finished` line.
