#!/bin/bash
#SBATCH --job-name=signclip-eval-popsign-zs-ytasl-sweep
#SBATCH --partition=standard
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -uo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
CONFIG_PATH="${CONFIG_PATH:-/home/faxu/multimodalhugs/configs/signclip_eval_popsign_zeroshot_youtube_asl_checkpoint_sweep.server.yaml}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"
EVAL_ROOT="${EVAL_ROOT:-/home/faxu/scratch/signclip/evals/popsign_zeroshot_youtube_asl_checkpoint_sweep}"

CHECKPOINTS=(
  "/home/faxu/scratch/signclip/runs/youtube_asl_pretrain_v2_multigpu/train/checkpoint-3000"
  "/home/faxu/scratch/signclip/runs/youtube_asl_pretrain_v2_multigpu/train/checkpoint-6000"
  "/home/faxu/scratch/signclip/runs/youtube_asl_pretrain_v2_multigpu/train/checkpoint-9000"
  "/home/faxu/scratch/signclip/runs/youtube_asl_pretrain_v1_resume_16096_plus8ep/train/checkpoint-40240"
  "/home/faxu/scratch/signclip/runs/youtube_asl_pretrain_v1_resume_16096_plus8ep/train/checkpoint-44264"
)

LABELS=(
  "multigpu-checkpoint-3000"
  "multigpu-checkpoint-6000"
  "multigpu-checkpoint-9000"
  "singlegpu-checkpoint-40240"
  "singlegpu-checkpoint-44264"
)

mkdir -p "${LOGS_ROOT}" "${EVAL_ROOT}"
cd "${REPO_PATH}"

echo "[eval-sweep] host=$(hostname)"
echo "[eval-sweep] repo=${REPO_PATH}"
echo "[eval-sweep] config=${CONFIG_PATH}"
echo "[eval-sweep] eval_root=${EVAL_ROOT}"
echo "[eval-sweep] start=$(date -Iseconds)"

nvidia-smi || true

failures=()

for index in "${!CHECKPOINTS[@]}"; do
  checkpoint="${CHECKPOINTS[$index]}"
  label="${LABELS[$index]}"
  output_dir="${EVAL_ROOT}/${label}"

  echo "[eval-sweep] label=${label}"
  echo "[eval-sweep] checkpoint=${checkpoint}"
  echo "[eval-sweep] output_dir=${output_dir}"
  echo "[eval-sweep] model_start=$(date -Iseconds)"

  if [[ ! -f "${checkpoint}/model.safetensors" ]]; then
    echo "[eval-sweep] ERROR: missing model.safetensors in ${checkpoint}" >&2
    failures+=("${label}:missing-checkpoint")
    continue
  fi

  if pixi run python -m multimodalhugs.tasks.contrastive.contrastive_training \
    --config_path "${CONFIG_PATH}" \
    --model_name_or_path "${checkpoint}" \
    --output_dir "${output_dir}" \
    --run_name "signclip-eval-popsign-zs-${label}" \
    --wandb_tags "signclip,popsign,eval,zeroshot,youtube-asl,server,v2t,checkpoint-sweep,${label}"; then
    echo "[eval-sweep] model_done=$(date -Iseconds) label=${label}"
  else
    echo "[eval-sweep] ERROR: evaluation failed for ${label}" >&2
    failures+=("${label}:evaluation-failed")
  fi
done

echo "[eval-sweep] done=$(date -Iseconds)"

if ((${#failures[@]} > 0)); then
  printf '[eval-sweep] failures=%s\n' "${failures[*]}" >&2
  exit 1
fi
