#!/bin/bash
#SBATCH --job-name=signclip-pretrain-popsign-smoke
#SBATCH --partition=standard
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
CONFIG_PATH="${CONFIG_PATH:-/home/faxu/multimodalhugs/configs/signclip_pretrain_popsign.server.yaml}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-/home/faxu/scratch/signclip/runs/popsign_pretrain_smoke_v1}"

mkdir -p "${LOGS_ROOT}"

cd "${REPO_PATH}"

echo "[smoke-train] host=$(hostname)"
echo "[smoke-train] repo=${REPO_PATH}"
echo "[smoke-train] config=${CONFIG_PATH}"
echo "[smoke-train] output_dir=${SMOKE_OUTPUT_DIR}"
echo "[smoke-train] start=$(date -Iseconds)"

nvidia-smi || true

pixi run python -m multimodalhugs.tasks.contrastive.contrastive_training \
  --config_path "${CONFIG_PATH}" \
  --output_dir "${SMOKE_OUTPUT_DIR}" \
  --max_steps 20 \
  --num_train_epochs 1 \
  --logging_steps 5 \
  --eval_strategy no \
  --save_strategy no \
  --run_retrieval_eval false

echo "[smoke-train] done=$(date -Iseconds)"
