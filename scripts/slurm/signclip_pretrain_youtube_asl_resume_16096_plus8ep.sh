#!/bin/bash
#SBATCH --job-name=signclip-pretrain-youtube-asl-r16096-p8
#SBATCH --partition=standard
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
CONFIG_PATH="${CONFIG_PATH:-/home/faxu/multimodalhugs/configs/signclip_pretrain_youtube_asl_resume_16096_plus8ep.server.yaml}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"

mkdir -p "${LOGS_ROOT}"

cd "${REPO_PATH}"

echo "[train-resume] host=$(hostname)"
echo "[train-resume] repo=${REPO_PATH}"
echo "[train-resume] config=${CONFIG_PATH}"
echo "[train-resume] start=$(date -Iseconds)"

nvidia-smi || true

pixi run python -m multimodalhugs.tasks.contrastive.contrastive_training \
  --config_path "${CONFIG_PATH}"

echo "[train-resume] done=$(date -Iseconds)"
