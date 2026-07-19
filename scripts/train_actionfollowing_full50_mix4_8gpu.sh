#!/usr/bin/env bash
set -euo pipefail

cd /mnt/gyc/cosmos-predict2.5
[[ -f .env ]] && source .env
source /mnt/gyc/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-predict2.5
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

export AFD_ROOT="${AFD_ROOT:-/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingData_LeRobot_Rot6D}"
export AFD_PROTOCOL="mix4"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/mnt/gyc/Action-Following/outputs/cosmos_predict25}"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE

[[ -f "$AFD_ROOT/exploration/tasks/turn_switch/meta/info.json" ]] || {
  echo "ERROR: canonical ActionFollowingData mix4 root is not mounted: $AFD_ROOT" >&2
  exit 1
}

torchrun --nproc_per_node=8 --master_port="${MASTER_PORT:-12347}" \
  -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  -- experiment=cosmos_predict25_actionfollowing_rot6d20 \
  job.name=cosmos_predict25_afd_full50_mix4_rot6d20_a32_bs16_40k \
  trainer.max_iter=40000
