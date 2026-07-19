#!/usr/bin/env bash
set -Eeuo pipefail

die() {
  echo "[FATAL] $*" >&2
  exit 1
}

ensure_mount_alias() {
  local alias_path="$1"
  local target_path="$2"
  [[ -d "$target_path" ]] || die "missing AIHC mount: $target_path"
  if [[ ! -e "$alias_path" ]]; then
    ln -s "$target_path" "$alias_path"
  fi
  [[ "$(readlink -f "$alias_path")" == "$(readlink -f "$target_path")" ]] || \
    die "$alias_path does not resolve to $target_path"
}

ensure_file_alias() {
  local alias_path="$1"
  local target_path="$2"
  [[ -f "$target_path" ]] || die "missing file alias target: $target_path"
  if [[ ! -e "$alias_path" ]]; then
    ln -s "$target_path" "$alias_path"
  fi
  [[ "$(readlink -f "$alias_path")" == "$(readlink -f "$target_path")" ]] || \
    die "$alias_path does not resolve to $target_path"
}

ensure_mount_alias /mnt/gyc /mnt/dataset/csx_workspace
ensure_mount_alias /mnt/gyc_ckp /mnt/dataset/csx_ckp
ensure_mount_alias /mnt/public_ckp /mnt/dataset/public_data

REPO=/mnt/gyc/cosmos-predict2.5
AFD_ROOT=/mnt/dataset/sixiangchen_workspace/Ideas/data/ActionFollowingData_LeRobot_Rot6D/train
C25_ACTION_CHECKPOINT_PATH=/mnt/dataset/public_data/cscsx_projects/cosmospredict2.5_infer/models/Cosmos-Predict2.5-2B/robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt
C25_TOKENIZER_PATH=/mnt/dataset/public_data/cscsx_projects/cosmospredict2.5_infer/models/Cosmos-Predict2.5-2B/tokenizer.pth
C25_REASON_PATH=/mnt/dataset/public_data/cscsx_projects/cosmospredict2.5_infer/models/Cosmos-Reason1-7B
OUT_BASE=/mnt/gyc_ckp/Action-Following/outputs/cosmos_predict25/mix4/smoke_20260718_retry12
C25_REASON_PROCESSOR_PATH="$OUT_BASE/reason1_processor"

[[ -f "$AFD_ROOT/demo_clean_zed2i_visible/turn_switch/meta/info.json" ]] || die "clean data missing"
[[ -f "$AFD_ROOT/exploration/tasks/turn_switch/meta/info.json" ]] || die "mix4 exploration data missing"
[[ -f "$C25_ACTION_CHECKPOINT_PATH" ]] || die "Predict2.5 action-conditioned checkpoint missing"
[[ -f "$C25_TOKENIZER_PATH" ]] || die "Predict2.5 tokenizer missing: $C25_TOKENIZER_PATH"
[[ -f "$C25_REASON_PATH/config.json" ]] || die "Predict2.5 Reason1 config missing: $C25_REASON_PATH"
[[ -f "$C25_REASON_PATH/tokenizer.json" ]] || die "Predict2.5 Reason1 tokenizer missing: $C25_REASON_PATH"
[[ -f "$C25_REASON_PATH/model.safetensors.index.json" ]] || die "Predict2.5 Reason1 weights index missing: $C25_REASON_PATH"
[[ -f "$REPO/cosmos_predict2/_src/reason1/tokenizer/preprocessor_config.json" ]] || \
  die "Qwen2.5-VL processor config missing from adapter"

mkdir -p "$C25_REASON_PROCESSOR_PATH"
for processor_file in chat_template.json config.json tokenizer.json tokenizer_config.json; do
  ensure_file_alias "$C25_REASON_PROCESSOR_PATH/$processor_file" "$C25_REASON_PATH/$processor_file"
done
ensure_file_alias \
  "$C25_REASON_PROCESSOR_PATH/preprocessor_config.json" \
  "$REPO/cosmos_predict2/_src/reason1/tokenizer/preprocessor_config.json"

export AFD_ROOT C25_ACTION_CHECKPOINT_PATH C25_TOKENIZER_PATH C25_REASON_PATH C25_REASON_PROCESSOR_PATH OUT_BASE
export AFD_VIDEO_FALLBACK_ROOTS=/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingData_LeRobot_Rot6D
export AFD_VIDEO_SYMLINK_PREFIX_REMAP=/mnt/dataset/csx_workspace/Ideas/data=/mnt/dataset/sixiangchen_workspace/Ideas/data
export AFD_PROTOCOL=mix4
export IMAGINAIRE_OUTPUT_ROOT="$OUT_BASE"
export LD_LIBRARY_PATH=
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=8
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE

source /mnt/gyc/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-predict2.5
if [[ -f "$REPO/.venv/bin/activate" ]]; then
  source "$REPO/.venv/bin/activate"
fi

echo "[PRECHECK] job=${AIHC_JOB_NAME:-UNKNOWN} gpus=${TRAINING_CARD_SIZE:-UNKNOWN}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "[DATA] auditing clean and mix4 on the real AIHC mount"
cd "$REPO"
python - <<'PY'
import gc
import json
import os

import torch

from cosmos_predict2._src.predict2.action.datasets.actionfollowing_rot6d20 import (
    ActionFollowingRot6D20Dataset,
)

expected = {
    "clean": 475_122,
    "perturbed": 250_000,
    "random_feasible": 1_350_000,
    "counterfactual_replay": 474_645,
    "exploration": 121_071,
}

for protocol in ("clean", "mix4"):
    dataset = ActionFollowingRot6D20Dataset(
        root=os.environ["AFD_ROOT"],
        protocol=protocol,
        device_payload=False,
        audit_num_samples=100_000,
        audit_max_abs_error=0.02,
    )
    wanted = {"clean": expected["clean"]} if protocol == "clean" else expected
    assert dataset.family_effective_counts == wanted, (dataset.family_effective_counts, wanted)
    assert len(set(dataset.task_by_source)) == 50
    audit = dataset.audit_sampling(num_samples=100_000, seed=20260717)
    assert audit["max_abs_error"] <= 0.02, audit
    item = dataset[0]
    assert tuple(item["action"].shape) == (32, 20), item["action"].shape
    assert item["video"].shape[1] == 33, item["video"].shape
    assert torch.isfinite(item["action"]).all()
    assert torch.isfinite(item["video"].float()).all()
    print(json.dumps({
        "protocol": protocol,
        "tasks": len(set(dataset.task_by_source)),
        "effective_counts": dataset.family_effective_counts,
        "audit": audit,
        "action_shape": list(item["action"].shape),
        "video_shape": list(item["video"].shape),
        "views": ["cam_high"],
    }, sort_keys=True))
    family_probe_indices = {}
    for sample_index in range(min(len(dataset), 10_000)):
        record_index, _ = dataset._weighted_record_offset(sample_index)
        family = dataset.record_families[record_index]
        family_probe_indices.setdefault(family, sample_index)
        if set(family_probe_indices) == set(wanted):
            break
    assert set(family_probe_indices) == set(wanted), family_probe_indices
    for family, sample_index in sorted(family_probe_indices.items()):
        family_item = dataset[sample_index]
        assert family_item["family"] == family, (family_item["family"], family)
        assert tuple(family_item["action"].shape) == (32, 20)
        assert family_item["video"].shape[1] == 33
        print(json.dumps({
            "decode_probe_family": family,
            "sample_index": sample_index,
            "action_shape": list(family_item["action"].shape),
            "video_shape": list(family_item["video"].shape),
        }, sort_keys=True))
        del family_item
    del dataset, item
    gc.collect()
PY

run_train() {
  local per_rank_batch="$1"
  local label="$2"
  local master_port="$3"
  local output_root="$OUT_BASE/$label"
  local run_name="cosmos_predict25_afd_full50_mix4_rot6d20_a32_${label}_20step_smoke"
  mkdir -p "$output_root"
  echo "[TRAIN] label=$label per_rank_batch=$per_rank_batch global_batch=$((per_rank_batch * 8))"
  IMAGINAIRE_OUTPUT_ROOT="$output_root" \
  torchrun --nproc_per_node=8 --master_port="$master_port" \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment=cosmos_predict25_actionfollowing_rot6d20 \
    job.name="$run_name" \
    trainer.max_iter=20 \
    trainer.logging_iter=1 \
    checkpoint.save_iter=20 \
    checkpoint.load_path="$C25_ACTION_CHECKPOINT_PATH" \
    +model.config.tokenizer.vae_pth="$C25_TOKENIZER_PATH" \
    model.config.text_encoder_config.ckpt_path="$C25_REASON_PATH" \
    +model.config.text_encoder_config.model_config.tokenizer.cache_dir="$C25_REASON_PROCESSOR_PATH" \
    dataloader_train.batch_size="$per_rank_batch" \
    2>&1 | tee "$output_root/train.log"
}

if run_train 2 bs16 29551; then
  EFFECTIVE_BATCH=16
  TRAIN_OUT="$OUT_BASE/bs16"
else
  rc=$?
  if grep -RqiE "CUDA out of memory|CUDA error: out of memory|OutOfMemoryError" "$OUT_BASE/bs16"; then
    echo "[OOM] global batch 16 failed; retrying the allowed global batch 8 fallback"
    run_train 1 bs8 29552
    EFFECTIVE_BATCH=8
    TRAIN_OUT="$OUT_BASE/bs8"
  else
    exit "$rc"
  fi
fi

LATEST_FILE="$(find "$TRAIN_OUT" -name latest_checkpoint.txt -type f -print -quit)"
[[ -n "$LATEST_FILE" ]] || die "20-step run completed without latest_checkpoint.txt"
LATEST_ITER="$(cat "$LATEST_FILE")"
[[ -e "$(dirname "$LATEST_FILE")/$LATEST_ITER" ]] || die "latest checkpoint missing"

printf 'status=passed\nmodel=Cosmos-Predict2.5-2B\nprotocol=mix4\nsteps=20\neffective_global_batch=%s\ncheckpoint=%s\n' \
  "$EFFECTIVE_BATCH" "$(dirname "$LATEST_FILE")/$LATEST_ITER" | tee "$OUT_BASE/SMOKE_RESULT.txt"
