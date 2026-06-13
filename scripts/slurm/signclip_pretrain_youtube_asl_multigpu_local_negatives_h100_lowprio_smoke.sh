#!/bin/bash
#SBATCH --job-name=signclip-ytasl-mgpu-localneg-h100-lowprio
#SBATCH --partition=lowprio
#SBATCH --gres=gpu:H100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
CONFIG_PATH="${CONFIG_PATH:-/home/faxu/multimodalhugs/configs/signclip_pretrain_youtube_asl_multigpu_local_negatives_h100_lowprio_smoke.server.yaml}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

mkdir -p "${LOGS_ROOT}"
cd "${REPO_PATH}"

echo "[localneg-h100-lowprio] host=$(hostname)"
echo "[localneg-h100-lowprio] config=${CONFIG_PATH}"
echo "[localneg-h100-lowprio] gpus_per_node=${GPUS_PER_NODE}"
echo "[localneg-h100-lowprio] expected_random_loss=ln(128)=4.852"
echo "[localneg-h100-lowprio] start=$(date -Iseconds)"

nvidia-smi || true

export ACCELERATE_USE_CPU=false
export TORCH_DISTRIBUTED_DEFAULT_BACKEND=nccl

pixi run torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}" \
  -m multimodalhugs.tasks.contrastive.contrastive_training \
  --config_path "${CONFIG_PATH}"

echo "[localneg-h100-lowprio] done=$(date -Iseconds)"
