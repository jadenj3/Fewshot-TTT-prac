#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/current/ttt_%A_%a.log
#SBATCH --nodelist=dogo
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --array=0-4

# ====================
# Environment Setup
# ====================
export HOME=/home/ubuntu
cd ~/Fewshot-TTT

# -------------------------------------------------------------------------
# Fixed parameters
# -------------------------------------------------------------------------
K_SHOT=10
VOTE_PERMUTATIONS=5
LORA_DROPOUT=0.05

DATE=$(date +"%Y%m%d")
LOG_DIR="logs/current"
MODEL_DIR="${HOME}/Fewshot-TTT-prac/meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_TYPE="masked_inputs_text_completion_dataset" # all outputs
# DATASET_TYPE="text_completion_dataset" # all inputs and outputs
# DATASET_TYPE="masked_text_completion_dataset" # last output
mkdir -p "${LOG_DIR}"

# -------------------------------------------------------------------------
# List all your experiment parameter sets
# -------------------------------------------------------------------------
EXPERIMENTS=(
  "main 1e-4 64 64 5 1 40 True False False 42"
  "main 1e-4 64 64 5 1 40 True False False 43"
  "main 1e-4 64 64 5 1 40 True False False 44"
  "main 1e-4 64 64 5 1 40 True False False 45"
  "main 1e-4 64 64 5 1 40 True False False 46"
)

# -------------------------------------------------------------------------
# Select the parameter set for THIS sub-job
# Slurm sets SLURM_ARRAY_TASK_ID automatically (0..9)
# -------------------------------------------------------------------------
PARAMS="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
read -r EXP_NOTE LR LORA_RANK LORA_ALPHA BATCH_SIZE EPOCHS NUM_TRAINING_STEPS \
       SHUFFLE MAJORITY_VOTE LEAVE_ONE_OUT SEED <<< "${PARAMS}"

# Convert your SEED if needed:
SEED_NUM=$(( SEED - 41 ))

# Construct a nice experiment name
EXP_NAME="TTT_${EXP_NOTE}_${K_SHOT}_${DATASET_TYPE}_${NUM_TRAINING_STEPS}_${SHUFFLE}_${MAJORITY_VOTE}_${VOTE_PERMUTATIONS}_${LEAVE_ONE_OUT}_${BATCH_SIZE}_${EPOCHS}_${LR}_${LORA_RANK}_${LORA_ALPHA}_${LORA_DROPOUT}_${SEED_NUM}"
OUTPUT_FILE="${LOG_DIR}/${DATE}_${EXP_NAME}.json"
LOGFILE="${LOG_DIR}/${DATE}_${EXP_NAME}.log"

echo "----------------------------------------------------------"
echo "[SUB-JOB: $SLURM_ARRAY_TASK_ID] Launching Experiment:"
echo "  EXP_NAME = ${EXP_NAME}"
echo "----------------------------------------------------------"

# Run your Python experiment
python3 -m src.methods.ttt \
  --exp_name "${EXP_NAME}" \
  --model_dir "${MODEL_DIR}" \
  --output_file "${OUTPUT_FILE}" \
  --k "${K_SHOT}" \
  --dataset_type "${DATASET_TYPE}" \
  --num_training_steps "${NUM_TRAINING_STEPS}" \
  --shuffle "${SHUFFLE}" \
  --majority_vote "${MAJORITY_VOTE}" \
  --vote_permutations "${VOTE_PERMUTATIONS}" \
  --leave_one_out "${LEAVE_ONE_OUT}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --seed "${SEED}" \
  > "${LOGFILE}" 2>&1
