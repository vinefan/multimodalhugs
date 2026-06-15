#!/bin/bash
#SBATCH --job-name=signclip-eval-popsign-zs-ytasl-v3
#SBATCH --partition=standard
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -uo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
CONFIG_PATH="${CONFIG_PATH:-/home/faxu/multimodalhugs/configs/signclip_eval_popsign_zeroshot_youtube_asl_checkpoint_sweep.server.yaml}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/faxu/scratch/signclip/runs/youtube_asl_pretrain_v3_10ep/train}"
EVAL_ROOT="${EVAL_ROOT:-/home/faxu/scratch/signclip/evals/popsign_zeroshot_youtube_asl_v3_10ep_sweep}"

mkdir -p "${LOGS_ROOT}" "${EVAL_ROOT}"
cd "${REPO_PATH}"

mapfile -t checkpoints < <(
  find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "checkpoint-*" | sort -V
)

if ((${#checkpoints[@]} == 0)); then
  echo "[v3-10ep-sweep] ERROR: no checkpoints found under ${CHECKPOINT_ROOT}" >&2
  exit 1
fi

echo "[v3-10ep-sweep] host=$(hostname)"
echo "[v3-10ep-sweep] checkpoint_root=${CHECKPOINT_ROOT}"
echo "[v3-10ep-sweep] checkpoint_count=${#checkpoints[@]}"
printf '[v3-10ep-sweep] checkpoint=%s\n' "${checkpoints[@]}"
echo "[v3-10ep-sweep] start=$(date -Iseconds)"

nvidia-smi || true

failures=()

for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name="$(basename "${checkpoint}")"
  label="v3-10ep-${checkpoint_name}"
  output_dir="${EVAL_ROOT}/${label}"

  echo "[v3-10ep-sweep] label=${label}"
  echo "[v3-10ep-sweep] checkpoint=${checkpoint}"
  echo "[v3-10ep-sweep] model_start=$(date -Iseconds)"

  if [[ ! -f "${checkpoint}/model.safetensors" ]]; then
    echo "[v3-10ep-sweep] ERROR: missing model.safetensors in ${checkpoint}" >&2
    failures+=("${label}:missing-checkpoint")
    continue
  fi

  if pixi run python -m multimodalhugs.tasks.contrastive.contrastive_training \
    --config_path "${CONFIG_PATH}" \
    --model_name_or_path "${checkpoint}" \
    --output_dir "${output_dir}" \
    --run_name "signclip-eval-popsign-zs-${label}" \
    --wandb_tags "signclip,popsign,eval,zeroshot,youtube-asl,server,v2t,checkpoint-sweep,v3,10ep,${checkpoint_name}"; then
    echo "[v3-10ep-sweep] model_done=$(date -Iseconds) label=${label}"
  else
    echo "[v3-10ep-sweep] ERROR: evaluation failed for ${label}" >&2
    failures+=("${label}:evaluation-failed")
  fi
done

echo "[v3-10ep-sweep] done=$(date -Iseconds)"

if ((${#failures[@]} > 0)); then
  printf '[v3-10ep-sweep] failures=%s\n' "${failures[*]}" >&2
  exit 1
fi
