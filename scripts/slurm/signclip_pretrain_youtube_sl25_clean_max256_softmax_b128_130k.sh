#!/bin/bash
#SBATCH --job-name=signclip-yt-sl25-b128
#SBATCH --partition=standard
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
CONFIG_PATH="${CONFIG_PATH:-/home/faxu/multimodalhugs/configs/signclip_pretrain_youtube_sl25_clean_max256_softmax_b128_130k.server.yaml}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"
PIXI_BIN="${PIXI_BIN:-/home/faxu/.pixi/bin/pixi}"

mkdir -p "${LOGS_ROOT}"

cd "${REPO_PATH}"

echo "[train-sl25-b128] host=$(hostname)"
echo "[train-sl25-b128] repo=${REPO_PATH}"
echo "[train-sl25-b128] config=${CONFIG_PATH}"
echo "[train-sl25-b128] start=$(date -Iseconds)"

nvidia-smi || true

"${PIXI_BIN}" run python -m multimodalhugs.tasks.contrastive.contrastive_training \
  --config_path "${CONFIG_PATH}"

echo "[train-sl25-b128] done=$(date -Iseconds)"
