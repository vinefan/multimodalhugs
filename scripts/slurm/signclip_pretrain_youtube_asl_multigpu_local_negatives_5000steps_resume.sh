#!/bin/bash
#SBATCH --job-name=signclip-ytasl-mgpu-localneg-5k-resume
#SBATCH --partition=standard
#SBATCH --gres=gpu:H100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
CONFIG_PATH="${CONFIG_PATH:-/home/faxu/multimodalhugs/configs/signclip_pretrain_youtube_asl_multigpu_local_negatives_5000steps.server.yaml}"
RUN_ROOT="${RUN_ROOT:-/home/faxu/scratch/signclip/runs/youtube_asl_pretrain_multigpu_local_negatives_5000steps}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

mkdir -p "${LOGS_ROOT}"
cd "${REPO_PATH}"

LATEST_CHECKPOINT="$(
  find "${RUN_ROOT}/train" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
    | sort -V \
    | tail -n 1
)"

if [[ -z "${LATEST_CHECKPOINT}" ]]; then
  echo "[localneg-5k-resume] ERROR: no checkpoint found under ${RUN_ROOT}/train" >&2
  exit 1
fi

echo "[localneg-5k-resume] host=$(hostname)"
echo "[localneg-5k-resume] config=${CONFIG_PATH}"
echo "[localneg-5k-resume] run_root=${RUN_ROOT}"
echo "[localneg-5k-resume] resume_from_checkpoint=${LATEST_CHECKPOINT}"
echo "[localneg-5k-resume] gpus_per_node=${GPUS_PER_NODE}"
echo "[localneg-5k-resume] target_total_max_steps=5000"
echo "[localneg-5k-resume] expected_random_loss=ln(128)=4.852"
echo "[localneg-5k-resume] start=$(date -Iseconds)"

nvidia-smi || true

export ACCELERATE_USE_CPU=false
export TORCH_DISTRIBUTED_DEFAULT_BACKEND=nccl

pixi run torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}" \
  -m multimodalhugs.tasks.contrastive.contrastive_training \
  --config_path "${CONFIG_PATH}" \
  --resume_from_checkpoint "${LATEST_CHECKPOINT}" \
  --run_name "signclip-pretrain-youtube-asl-multigpu-local-negatives-5000steps-resume"

echo "[localneg-5k-resume] done=$(date -Iseconds)"
