#!/bin/bash
#SBATCH --job-name=signclip-pretrain-youtube-asl-multigpu
#SBATCH --partition=standard
#SBATCH --gres=gpu:H100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
CONFIG_PATH="${CONFIG_PATH:-/home/faxu/multimodalhugs/configs/signclip_pretrain_youtube_asl_multigpu.server.yaml}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

mkdir -p "${LOGS_ROOT}"

cd "${REPO_PATH}"

echo "[multigpu-train] host=$(hostname)"
echo "[multigpu-train] repo=${REPO_PATH}"
echo "[multigpu-train] config=${CONFIG_PATH}"
echo "[multigpu-train] gpus_per_node=${GPUS_PER_NODE}"
echo "[multigpu-train] start=$(date -Iseconds)"

nvidia-smi || true

pixi run torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}" \
  -m multimodalhugs.tasks.contrastive.contrastive_training \
  --config_path "${CONFIG_PATH}"

echo "[multigpu-train] done=$(date -Iseconds)"
