#!/bin/bash
#SBATCH --job-name=signclip-popsign-knn10
#SBATCH --partition=standard
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Slurm executes a copied script from /var/spool. SLURM_SUBMIT_DIR retains the
# directory from which sbatch was invoked; the fallback supports direct runs.
DEFAULT_REPO_PATH="${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
REPO_PATH="${REPO_PATH:-${DEFAULT_REPO_PATH}}"
PORTABLE_ROOT="${PORTABLE_ROOT:-/home/faxu/signclip_eval_models/signclip_global512_step4000_portable}"
DATASET_DIR="${DATASET_DIR:-/home/faxu/scratch/signclip/setup/popsign_pretrain_v1/setup/datasets/default}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/faxu/scratch/signclip/evals/popsign_knn10_global512_step4000}"

mkdir -p "${OUTPUT_DIR}" /home/faxu/scratch/signclip/logs
cd "${REPO_PATH}"
export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[knn10] host=$(hostname)"
echo "[knn10] dataset=${DATASET_DIR}"
echo "[knn10] models=${PORTABLE_ROOT}"
echo "[knn10] output=${OUTPUT_DIR}"

pixi run python scripts/evaluation/signclip_few_shot_knn.py \
  --dataset-dir "${DATASET_DIR}" \
  --processor-path "${PORTABLE_ROOT}/processor" \
  --model "softmax=${PORTABLE_ROOT}/softmax_global512_step4000" \
  --model "ring_sigmoid=${PORTABLE_ROOT}/ring_sigmoid_global512_step4000" \
  --output-dir "${OUTPUT_DIR}" \
  --shots 10 \
  --seed 42 \
  --protocol both \
  --batch-size 128 \
  --num-workers 4
