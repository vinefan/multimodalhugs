#!/bin/bash
#SBATCH --job-name=signclip-ytasl-v100-localneg-b128
#SBATCH --partition=lowprio
#SBATCH --gres=gpu:V100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
CONFIG_PATH="${CONFIG_PATH:-/home/faxu/multimodalhugs/configs/signclip_pretrain_youtube_asl_multigpu_local_negatives_v100_batch128.server.yaml}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

mkdir -p "${LOGS_ROOT}"
cd "${REPO_PATH}"

echo "[v100-localneg-b128] host=$(hostname)"
echo "[v100-localneg-b128] config=${CONFIG_PATH}"
echo "[v100-localneg-b128] gpus_per_node=${GPUS_PER_NODE}"
echo "[v100-localneg-b128] expected_train_random_loss=ln(128)=4.852"
echo "[v100-localneg-b128] expected_eval_random_loss=ln(64)=4.159"
echo "[v100-localneg-b128] start=$(date -Iseconds)"

nvidia-smi || true

export ACCELERATE_USE_CPU=false
export TORCH_DISTRIBUTED_DEFAULT_BACKEND=nccl

pixi run torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}" \
  -m multimodalhugs.tasks.contrastive.contrastive_training \
  --config_path "${CONFIG_PATH}"

echo "[v100-localneg-b128] done=$(date -Iseconds)"
