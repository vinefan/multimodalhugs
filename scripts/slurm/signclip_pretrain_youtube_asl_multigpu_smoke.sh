#!/bin/bash
#SBATCH --job-name=signclip-pretrain-youtube-asl-mgpu-smoke
#SBATCH --partition=standard
#SBATCH --gres=gpu:H100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
CONFIG_PATH="${CONFIG_PATH:-/home/faxu/multimodalhugs/configs/signclip_pretrain_youtube_asl_multigpu.server.yaml}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-/home/faxu/scratch/signclip/runs/youtube_asl_pretrain_v2_multigpu_smoke}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

mkdir -p "${LOGS_ROOT}"

cd "${REPO_PATH}"

echo "[multigpu-smoke] host=$(hostname)"
echo "[multigpu-smoke] repo=${REPO_PATH}"
echo "[multigpu-smoke] config=${CONFIG_PATH}"
echo "[multigpu-smoke] output_dir=${SMOKE_OUTPUT_DIR}"
echo "[multigpu-smoke] gpus_per_node=${GPUS_PER_NODE}"
echo "[multigpu-smoke] start=$(date -Iseconds)"

nvidia-smi || true

pixi run torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}" \
  -m multimodalhugs.tasks.contrastive.contrastive_training \
  --config_path "${CONFIG_PATH}" \
  --output_dir "${SMOKE_OUTPUT_DIR}" \
  --max_steps 50 \
  --logging_steps 5 \
  --eval_strategy no \
  --save_strategy no \
  --run_retrieval_eval false

echo "[multigpu-smoke] done=$(date -Iseconds)"
