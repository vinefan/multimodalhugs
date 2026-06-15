#!/bin/bash
#SBATCH --job-name=signclip-ytasl-fixed-loss-probe-v2
#SBATCH --partition=standard
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
CONFIG_PATH="${CONFIG_PATH:-/home/faxu/multimodalhugs/configs/signclip_pretrain_youtube_asl_fixed_loss_probe.server.yaml}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"

mkdir -p "${LOGS_ROOT}"
cd "${REPO_PATH}"

echo "[fixed-loss-probe-v2] host=$(hostname)"
echo "[fixed-loss-probe-v2] config=${CONFIG_PATH}"
echo "[fixed-loss-probe-v2] start=$(date -Iseconds)"

nvidia-smi || true

pixi run python -m multimodalhugs.tasks.contrastive.contrastive_training \
  --config_path "${CONFIG_PATH}"

echo "[fixed-loss-probe-v2] done=$(date -Iseconds)"
