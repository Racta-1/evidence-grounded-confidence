#!/bin/bash
# run_groundcheck.sh: run GroundCheck models as a batch job.
#
# Plain-language summary: runs one or more models over the suite, one after
# another, on ONE GPU, then scores the results and draws the figures. Each run
# saves as it goes, so if the job stops, submitting it again picks up where it left off.
#
# With the Slurm scheduler (if `which sbatch` prints a path):
#   sbatch scripts/run_groundcheck.sh                               # all six models
#   sbatch scripts/run_groundcheck.sh qwen2_5_vl_7b --limit 20      # timed test: one model, 20 questions per dataset
#   sbatch scripts/run_groundcheck.sh medgemma_4b gemma3_4b         # chosen models
#
# Without Slurm (a shared GPU machine): runs in the background and survives logout:
#   nohup bash scripts/run_groundcheck.sh qwen2_5_vl_7b --limit 20 > logs/test.log 2>&1 &
#   tail -f logs/test.log          # watch progress (Ctrl+C stops watching, not the job)
#
# Before the first run: python code/01_fetch.py, code/01b_fetch_mimic.py (MIMIC, interactive:
# it asks for your password), and code/02_build_suite.py. A batch job cannot type a password.
#
#SBATCH --job-name=groundcheck
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
# If your cluster requires a partition, remove the first "#" on the next line and fill in its name:
##SBATCH --partition=PARTITION_NAME

set -euo pipefail

# --- go to the project folder ------------------------------------------------
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR"; else cd "$(dirname "$0")/.."; fi
mkdir -p logs

# --- model downloads go to the project's disk, not the (small) home folder ----
# about 80 GB for all six models; your Hugging Face login token stays where it is
export HF_HUB_CACHE="${HF_HUB_CACHE:-$(pwd)/.hf_cache/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-$(pwd)/.hf_cache/xet}"
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE"

# --- which models -------------------------------------------------------------
ALL_MODELS="qwen2_5_vl_7b lingshu_7b gemma3_4b medgemma_4b llava16_mistral_7b llavamed15_mistral_7b"
MODELS=()
EXTRA=()
for arg in "$@"; do
  if [[ "$arg" == --* || "$arg" =~ ^[0-9]+$ ]]; then EXTRA+=("$arg"); else MODELS+=("$arg"); fi
done
[[ ${#MODELS[@]} -eq 0 ]] && read -r -a MODELS <<< "$ALL_MODELS"

# --- pick ONE GPU when not under Slurm (Slurm sets CUDA_VISIBLE_DEVICES itself) ---
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # free = least memory in use among GPUs under 10% utilization
  PICK=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
         | awk -F', ' '$3 < 10 {print $2, $1}' | sort -n | head -1 | awk '{print $2}')
  if [[ -z "$PICK" ]]; then
    echo "No idle GPU right now (all above 10% use). Try again later, or set CUDA_VISIBLE_DEVICES yourself." >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$PICK"
fi

echo "=== GroundCheck job started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname), GPU ${CUDA_VISIBLE_DEVICES}"
echo "models: ${MODELS[*]}   extra options: ${EXTRA[*]:-none}"
echo "model files stored in: $HF_HUB_CACHE ($(df -h "$HF_HUB_CACHE" | awk 'NR==2 {print $4}') free there)"
python -c "import torch, transformers; print('python ok | torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0), '| transformers', transformers.__version__)"
[[ -f data/suite_private/items.csv ]] && echo "MIMIC-CXR items found (private)" || echo "note: MIMIC-CXR items not built; MIMIC will be skipped"

for m in "${MODELS[@]}"; do
  echo "=== $m started $(date '+%H:%M:%S')"
  python code/04_run_model.py "configs/${m}.yaml" ${EXTRA[@]+"${EXTRA[@]}"} || echo "!!! $m stopped with an error; continuing with the next model"
  echo "=== $m ended $(date '+%H:%M:%S')"
done

python code/05_score.py
python code/06_figures.py
echo "=== GroundCheck job finished $(date '+%Y-%m-%d %H:%M:%S')"
