#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/xf/local/src/multimodalhugs-ring-experiment}"
LOG_DIR="${LOG_DIR:-/home/xf/local/scratch/signclip/logs/fingerclip_multigpu_bs16_25ep_compare}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.pixi/envs/default/bin/python}"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

run_experiment() {
  local name="$1"
  local config_path="$2"
  local log_path="${LOG_DIR}/${name}.log"

  printf '[%s] Starting %s\n' "$(date --iso-8601=seconds)" "${name}" | tee -a "${log_path}"
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}" \
    -m multimodalhugs.tasks.contrastive.contrastive_training \
    --config_path "${config_path}" 2>&1 | tee -a "${log_path}"
  printf '[%s] Completed %s\n' "$(date --iso-8601=seconds)" "${name}" | tee -a "${log_path}"
}

run_experiment \
  "fingerclip-multigpu-siglip-ring-bs16-25ep" \
  "configs/signclip_train_fingerclip_multigpu_siglip_ring_25ep.sxma.yaml"

run_experiment \
  "fingerclip-multigpu-softmax-global-bs16-25ep" \
  "configs/signclip_train_fingerclip_multigpu_softmax_global_25ep.sxma.yaml"
