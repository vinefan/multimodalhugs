#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/xf/data/src/multimodalhugs}"
RUN_ROOT="${RUN_ROOT:-/home/xf/data/signclip_pretrain/as1_softmax_b32_half_fps_truncate256_10ep}"
SETUP_CONFIG="${SETUP_CONFIG:-${REPO_PATH}/configs/signclip_setup_youtube_asl_half_fps_truncate256.as1.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_PATH}/configs/signclip_pretrain_youtube_asl_as1_softmax_b32_half_fps_truncate256_10ep.yaml}"
SOURCE_RUNNER="${SOURCE_RUNNER:-/home/xf/data/signclip_pretrain/as1_softmax_b32_10ep/run_contrastive_no_sdp.py}"

mkdir -p "${RUN_ROOT}/logs"
cd "${REPO_PATH}"

pixi run mmhugs-setup \
  --modality sign_clip \
  --config_path "${SETUP_CONFIG}" \
  --rebuild_dataset_from_scratch true

exec pixi run torchrun --standalone --nproc_per_node=1 \
  "${SOURCE_RUNNER}" \
  --config_path "${TRAIN_CONFIG}" \
  2>&1 | tee "${RUN_ROOT}/logs/train.log"
