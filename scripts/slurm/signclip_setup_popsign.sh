#!/bin/bash
#SBATCH --job-name=signclip-setup-popsign
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
CONFIG_PATH="${CONFIG_PATH:-/home/faxu/multimodalhugs/configs/signclip_setup_popsign.server.yaml}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"

mkdir -p "${LOGS_ROOT}"

cd "${REPO_PATH}"

echo "[setup] host=$(hostname)"
echo "[setup] repo=${REPO_PATH}"
echo "[setup] config=${CONFIG_PATH}"
echo "[setup] start=$(date -Iseconds)"

pixi run python -m multimodalhugs.multimodalhugs_cli.training_setup \
  --config_path "${CONFIG_PATH}"

echo "[setup] done=$(date -Iseconds)"
