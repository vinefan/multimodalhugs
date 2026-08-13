#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/xf/data/src/multimodalhugs}"
PIXI_BIN="${PIXI_BIN:-/home/xf/.local/bin/pixi}"
RUN_ROOT="${RUN_ROOT:-/home/xf/data/signclip_pretrain/as1_softmax_b32_half_fps_truncate256_10ep}"
SETUP_CONFIG="${SETUP_CONFIG:-${REPO_PATH}/configs/signclip_setup_youtube_asl_half_fps_truncate256.as1.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_PATH}/configs/signclip_pretrain_youtube_asl_as1_softmax_b32_half_fps_truncate256_10ep.yaml}"
SOURCE_RUNNER="${SOURCE_RUNNER:-/home/xf/data/signclip_pretrain/as1_softmax_b32_10ep/run_contrastive_no_sdp.py}"
SOURCE_METADATA_ROOT="${SOURCE_METADATA_ROOT:-/mnt/data/xf/datasets/YouTube-ASL/metadata_signclip_clean_v2}"
METADATA_ROOT="${METADATA_ROOT:-${RUN_ROOT}/metadata}"
POSE_ROOT="${POSE_ROOT:-/mnt/data/xf/datasets/YouTube-ASL}"

mkdir -p "${RUN_ROOT}/logs" "${METADATA_ROOT}"
cd "${REPO_PATH}"

for split in train validation test; do
  sed \
    -e "s#^/shares/iict-sp2.ebling.cl.uzh/common/YouTube-ASL/#${POSE_ROOT}/#" \
    -e "s#^/home/xf/data/datasets/YouTube-ASL/#${POSE_ROOT}/#" \
    "${SOURCE_METADATA_ROOT}/${split}.tsv" > "${METADATA_ROOT}/${split}.tsv"
done

"${PIXI_BIN}" run mmhugs-setup \
  --modality sign_clip \
  --config_path "${SETUP_CONFIG}" \
  --rebuild_dataset_from_scratch true

exec "${PIXI_BIN}" run torchrun --standalone --nproc_per_node=1 \
  "${SOURCE_RUNNER}" \
  --config_path "${TRAIN_CONFIG}" \
  2>&1 | tee "${RUN_ROOT}/logs/train.log"
