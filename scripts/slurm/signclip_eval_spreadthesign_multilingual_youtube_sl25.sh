#!/bin/bash
#SBATCH --job-name=signclip-eval-sts-multilingual
#SBATCH --partition=standard
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
CHECKPOINT="${CHECKPOINT:-/home/faxu/scratch/signclip/runs/youtube_sl25_clean_max256_softmax_b128_130k/train/checkpoint-36000}"
PROCESSOR="${PROCESSOR:-/home/faxu/scratch/signclip/setup/youtube_sl25_clean_max256_v1/setup/sign_clip_processor}"
METADATA_ROOT="${METADATA_ROOT:-/home/faxu/scratch/signclip/metadata/spreadthesign_multilingual_youtube_sl25}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/faxu/scratch/signclip/evals/spreadthesign_multilingual_zeroshot_youtube_sl25_checkpoint36000}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"
PIXI_BIN="${PIXI_BIN:-/home/faxu/.pixi/bin/pixi}"
FORCE_EVAL="${FORCE_EVAL:-0}"
PAIR_KEYS_OVERRIDE="${PAIR_KEYS_OVERRIDE:-en_ase en_ins pl_pso de_gsg en_bfi it_ise ja_jsl}"
read -r -a PAIR_KEYS <<< "${PAIR_KEYS_OVERRIDE}"

mkdir -p "${LOGS_ROOT}" "${RESULTS_ROOT}"
cd "${REPO_PATH}"

test -f "${CHECKPOINT}/model.safetensors"
test -d "${PROCESSOR}"
test -f "${METADATA_ROOT}/summary.json"

echo "[eval-sts-multilingual] host=$(hostname)"
echo "[eval-sts-multilingual] checkpoint=${CHECKPOINT}"
echo "[eval-sts-multilingual] processor=${PROCESSOR}"
echo "[eval-sts-multilingual] metadata_root=${METADATA_ROOT}"
echo "[eval-sts-multilingual] results_root=${RESULTS_ROOT}"
echo "[eval-sts-multilingual] pairs=${PAIR_KEYS[*]}"
echo "[eval-sts-multilingual] start=$(date -Iseconds)"

nvidia-smi || true

for pair in "${PAIR_KEYS[@]}"; do
  metadata_tsv="${METADATA_ROOT}/pairs/${pair}/test.tsv"
  result_path="${RESULTS_ROOT}/${pair}/eval_results.json"
  test -s "${metadata_tsv}"

  if [[ "${FORCE_EVAL}" != "1" && -s "${result_path}" ]]; then
    echo "[eval-sts-multilingual] skip completed pair=${pair} result=${result_path}"
    continue
  fi

  echo "[eval-sts-multilingual] pair=${pair} start=$(date -Iseconds)"
  "${PIXI_BIN}" run python scripts/evaluation/evaluate_signclip_v2t_fast.py \
    --checkpoint "${CHECKPOINT}" \
    --processor "${PROCESSOR}" \
    --metadata-tsv "${metadata_tsv}" \
    --output "${result_path}" \
    --batch-size 128 \
    --text-batch-size 256 \
    --score-chunk-size 256 \
    --num-workers 4
  echo "[eval-sts-multilingual] pair=${pair} done=$(date -Iseconds)"
done

summary_args=()
for pair in "${PAIR_KEYS[@]}"; do
  summary_args+=(--pair "${pair}")
done
"${PIXI_BIN}" run python scripts/evaluation/summarize_spreadthesign_multilingual.py \
  --results-root "${RESULTS_ROOT}" \
  "${summary_args[@]}"

echo "[eval-sts-multilingual] done=$(date -Iseconds)"
