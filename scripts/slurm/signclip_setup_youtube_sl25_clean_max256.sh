#!/bin/bash
#SBATCH --job-name=signclip-setup-yt-sl25
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
CONFIG_PATH="${CONFIG_PATH:-/home/faxu/multimodalhugs/configs/signclip_setup_youtube_sl25_clean_max256.server.yaml}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"

mkdir -p "${LOGS_ROOT}"

cd "${REPO_PATH}"

echo "[setup-sl25] host=$(hostname)"
echo "[setup-sl25] repo=${REPO_PATH}"
echo "[setup-sl25] config=${CONFIG_PATH}"
echo "[setup-sl25] start=$(date -Iseconds)"

pixi run python -m multimodalhugs.multimodalhugs_cli.training_setup \
  --config_path "${CONFIG_PATH}" \
  --modality sign_clip

echo "[setup-sl25] done=$(date -Iseconds)"
