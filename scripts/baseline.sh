#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/current/baseline_%A_%a.log
#SBATCH --nodelist=dogo
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --array=0-4

# ====================
# Environment Setup
# ====================
export HOME=/data/cl/u/adamz
source ~/.bashrc
conda activate tttenv
cd ~/Fewshot-TTT

# -------------------------------------------------------------------------
# Fixed/Shared Parameters
# -------------------------------------------------------------------------
DATE=$(date +"%Y%m%d")
LOG_DIR="logs/current"
MODEL_DIR="${HOME}/Models/Llama-3.1-8B-Instruct"

VOTE_PERMUTATIONS=5

mkdir -p "${LOG_DIR}"

# -------------------------------------------------------------------------
# Experiment parameter sets
# Each entry typically:
#   "exp K_SHOT MAJORITY_VOTE SEED"
# -------------------------------------------------------------------------
EXPERIMENTS=(
  "exp 10 False 42"
  "exp 10 False 43"
  "exp 10 False 44"
  "exp 10 False 45"
  "exp 10 False 46"
)

# -------------------------------------------------------------------------
# Select parameters for THIS sub-job using SLURM_ARRAY_TASK_ID
# -------------------------------------------------------------------------
PARAMS="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
read -r EXP_NOTE K_SHOT MAJORITY_VOTE SEED <<< "${PARAMS}"

# Construct a descriptive experiment name
EXP_NAME="Baseline_${EXP_NOTE}_${K_SHOT}_${SEED}"

# Output filenames
OUTPUT_FILE="${LOG_DIR}/${DATE}_${EXP_NAME}.json"
LOGFILE="${LOG_DIR}/${DATE}_${EXP_NAME}.log"

echo "----------------------------------------------------------"
echo "[SUB-JOB: $SLURM_ARRAY_TASK_ID] Launching Baseline Experiment:"
echo "  EXP_NAME = ${EXP_NAME}"
echo "----------------------------------------------------------"

# -------------------------------------------------------------------------
# Run the Python baseline experiment
# -------------------------------------------------------------------------
python3 -m src.methods.baseline \
  --exp_name "${EXP_NAME}" \
  --model_dir "${MODEL_DIR}" \
  --output_file "${OUTPUT_FILE}" \
  --task_start 0 \
  --task_end 27 \
  --k "${K_SHOT}" \
  --majority_vote "${MAJORITY_VOTE}" \
  --vote_permutations "${VOTE_PERMUTATIONS}" \
  --seed "${SEED}" \
  > "${LOGFILE}" 2>&1

echo "Baseline sub-job $SLURM_ARRAY_TASK_ID completed."
