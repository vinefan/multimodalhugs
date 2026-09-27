#!/bin/bash
#SBATCH --job-name=signclip-prep-sts-multilingual
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
SOURCE_CSV="${SOURCE_CSV:-/shares/iict-sp2.ebling.cl.uzh/common/spreadthesign/SperadTheSign.csv}"
POSE_ROOT="${POSE_ROOT:-/shares/iict-sp2.ebling.cl.uzh/common/spreadthesign/sign-mt-poses}"
METADATA_ROOT="${METADATA_ROOT:-/home/faxu/scratch/signclip/metadata/spreadthesign_multilingual_youtube_sl25}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"
PIXI_BIN="${PIXI_BIN:-/home/faxu/.pixi/bin/pixi}"
PAIR_KEYS_OVERRIDE="${PAIR_KEYS_OVERRIDE:-en_ase en_ins pl_pso de_gsg en_bfi it_ise ja_jsl}"
LIMIT_PER_PAIR="${LIMIT_PER_PAIR:-}"

read -r -a PAIR_KEYS <<< "${PAIR_KEYS_OVERRIDE}"
pair_args=()
for pair in "${PAIR_KEYS[@]}"; do
  text_language="${pair%%_*}"
  sign_language="${pair#*_}"
  if [[ -z "${text_language}" || -z "${sign_language}" || "${text_language}" == "${sign_language}" ]]; then
    echo "Invalid pair key: ${pair}; expected text_sign (for example en_ase)" >&2
    exit 2
  fi
  pair_args+=(--pair "${text_language}:${sign_language}")
done

limit_args=()
if [[ -n "${LIMIT_PER_PAIR}" ]]; then
  limit_args+=(--limit-per-pair "${LIMIT_PER_PAIR}")
fi

mkdir -p "${LOGS_ROOT}" "${METADATA_ROOT}"
cd "${REPO_PATH}"

echo "[prep-sts-multilingual] host=$(hostname)"
echo "[prep-sts-multilingual] source_csv=${SOURCE_CSV}"
echo "[prep-sts-multilingual] pose_root=${POSE_ROOT}"
echo "[prep-sts-multilingual] metadata_root=${METADATA_ROOT}"
echo "[prep-sts-multilingual] pairs=${PAIR_KEYS[*]}"
echo "[prep-sts-multilingual] limit_per_pair=${LIMIT_PER_PAIR:-none}"
echo "[prep-sts-multilingual] start=$(date -Iseconds)"

"${PIXI_BIN}" run python scripts/prepare_spreadthesign_multilingual_eval.py \
  --source-csv "${SOURCE_CSV}" \
  --pose-root "${POSE_ROOT}" \
  --output-dir "${METADATA_ROOT}" \
  --max-frames 256 \
  --num-workers 4 \
  --validate-poses \
  "${pair_args[@]}" \
  "${limit_args[@]}"

echo "[prep-sts-multilingual] done=$(date -Iseconds)"
