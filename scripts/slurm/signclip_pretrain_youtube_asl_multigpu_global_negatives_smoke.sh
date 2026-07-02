#!/bin/bash
#SBATCH --job-name=signclip-ytasl-mgpu-globalneg-smoke
#SBATCH --partition=standard
#SBATCH --gres=gpu:H100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
CONFIG_PATH="${CONFIG_PATH:-/home/faxu/multimodalhugs/configs/signclip_pretrain_youtube_asl_multigpu_global_negatives_smoke.server.yaml}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

mkdir -p "${LOGS_ROOT}"
cd "${REPO_PATH}"

echo "[globalneg-smoke] host=$(hostname)"
echo "[globalneg-smoke] config=${CONFIG_PATH}"
echo "[globalneg-smoke] gpus_per_node=${GPUS_PER_NODE}"
echo "[globalneg-smoke] expected_train_random_loss=ln(512)=6.238"
echo "[globalneg-smoke] expected_eval_random_loss=ln(256)=5.545"
echo "[globalneg-smoke] start=$(date -Iseconds)"

nvidia-smi || true

export ACCELERATE_USE_CPU=false
export TORCH_DISTRIBUTED_DEFAULT_BACKEND=nccl

pixi run torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}" \
  -m multimodalhugs.tasks.contrastive.contrastive_training \
  --config_path "${CONFIG_PATH}"

echo "[globalneg-smoke] done=$(date -Iseconds)"
