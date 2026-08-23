#!/bin/bash
#SBATCH --job-name=signclip-eval-popsign-zs-sl25-best
#SBATCH --partition=standard
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/faxu/scratch/signclip/logs/%x-%j.out
#SBATCH --error=/home/faxu/scratch/signclip/logs/%x-%j.err

set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/faxu/multimodalhugs}"
CONFIG_PATH="${CONFIG_PATH:-/home/faxu/multimodalhugs/configs/signclip_eval_popsign_zeroshot_youtube_sl25_current_best.server.yaml}"
TRAIN_DIR="${TRAIN_DIR:-/home/faxu/scratch/signclip/runs/youtube_sl25_clean_max256_softmax_b128_130k/train}"
EVAL_ROOT="${EVAL_ROOT:-/home/faxu/scratch/signclip/evals/popsign_zeroshot_youtube_sl25_current_best}"
LOGS_ROOT="${LOGS_ROOT:-/home/faxu/scratch/signclip/logs}"
PIXI_BIN="${PIXI_BIN:-/home/faxu/.pixi/bin/pixi}"

mkdir -p "${LOGS_ROOT}" "${EVAL_ROOT}"

cd "${REPO_PATH}"

BEST_CHECKPOINT="$("${PIXI_BIN}" run python - "${TRAIN_DIR}" <<'PY'
import json
import sys
from pathlib import Path

train_dir = Path(sys.argv[1])
state_path = train_dir / "trainer_state.json"
best = None
if state_path.exists():
    with state_path.open("r", encoding="utf-8") as handle:
        best = json.load(handle).get("best_model_checkpoint")

if best and (Path(best) / "model.safetensors").exists():
    print(best)
    raise SystemExit(0)

checkpoints = sorted(
    [p for p in train_dir.glob("checkpoint-*") if (p / "model.safetensors").exists()],
    key=lambda p: int(p.name.rsplit("-", 1)[-1]),
)
if not checkpoints:
    raise SystemExit(f"No usable checkpoints found under {train_dir}")
print(checkpoints[-1])
PY
)"

CHECKPOINT_NAME="$(basename "${BEST_CHECKPOINT}")"
OUTPUT_DIR="${EVAL_ROOT}/${CHECKPOINT_NAME}"

echo "[eval-popsign-sl25-best] host=$(hostname)"
echo "[eval-popsign-sl25-best] repo=${REPO_PATH}"
echo "[eval-popsign-sl25-best] config=${CONFIG_PATH}"
echo "[eval-popsign-sl25-best] train_dir=${TRAIN_DIR}"
echo "[eval-popsign-sl25-best] best_checkpoint=${BEST_CHECKPOINT}"
echo "[eval-popsign-sl25-best] output_dir=${OUTPUT_DIR}"
echo "[eval-popsign-sl25-best] start=$(date -Iseconds)"

nvidia-smi || true

"${PIXI_BIN}" run python -m multimodalhugs.tasks.contrastive.contrastive_training \
  --config_path "${CONFIG_PATH}" \
  --model_name_or_path "${BEST_CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "signclip-eval-popsign-zs-youtube-sl25-${CHECKPOINT_NAME}" \
  --wandb_tags "signclip,popsign,eval,zeroshot,youtube-sl25,current-best,v2t,${CHECKPOINT_NAME}"

echo "[eval-popsign-sl25-best] done=$(date -Iseconds)"
