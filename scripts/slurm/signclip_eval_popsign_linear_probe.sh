#!/bin/bash
#SBATCH --job-name=signclip-popsign-linear
#SBATCH --partition=standard
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
PORTABLE_ROOT="${PORTABLE_ROOT:-/home/faxu/signclip_eval_models/signclip_global512_step4000_portable}"
DATASET_DIR="${DATASET_DIR:-/home/faxu/scratch/signclip/setup/popsign_pretrain_v1/setup/datasets/default}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/faxu/scratch/signclip/evals/popsign_linear_global512_step4000}"

mkdir -p "${OUTPUT_DIR}" /home/faxu/scratch/signclip/logs
cd "${REPO_PATH}"
export PYTHONPATH="${REPO_PATH}"

echo "[linear] host=$(hostname)"
echo "[linear] repo=${REPO_PATH}"
echo "[linear] PYTHONPATH=${PYTHONPATH}"
echo "[linear] dataset=${DATASET_DIR}"
echo "[linear] models=${PORTABLE_ROOT}"
echo "[linear] output=${OUTPUT_DIR}"

pixi run python scripts/evaluation/signclip_linear_probe.py \
  --dataset-dir "${DATASET_DIR}" \
  --processor-path "${PORTABLE_ROOT}/processor" \
  --model "softmax=${PORTABLE_ROOT}/softmax_global512_step4000" \
  --model "ring_sigmoid=${PORTABLE_ROOT}/ring_sigmoid_global512_step4000" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 128 \
  --num-workers 4 \
  --bad-sample-policy skip
