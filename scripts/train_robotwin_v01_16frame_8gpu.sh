#!/bin/bash
# Action-Conditioned 后训练 V0.1 —— RoboTwin 双臂 14D, 生成 16 帧, 224×224
# 基于现有 AC 14D experiment 做参数 override：
#   - num_action_per_chunk: 12 → 16（生成帧数 = chunk size）
#   - video_size: [256, 320] → [224, 224]
#   - fps_downsample_ratio: 10 → 1（24 fps 不抽帧）
#   - state_t: 4 → 5（= 1 + 16/4, tokenizer 时间压缩 4×）
#   - max_iter: 150k → 100k
#   - action_dim: 14（沿用现有 endpose 14D 数据）

source .env

export IMAGINAIRE_OUTPUT_ROOT=/mnt/gyc_ckp/training_outputs

# NVIDIA 库补丁（容器内 NVML/驱动版本不匹配）
if [ -d /mnt/public_ckp/cscsx_projects/libnvidia-535 ]; then
    cd /mnt/public_ckp/cscsx_projects/libnvidia-535
    cp -r libnvidia-gl-535/* / 2>/dev/null || true
    cp -r libnvidia-compute-535/* / 2>/dev/null || true
    cp -r nvidia-utils/* / 2>/dev/null || true
    cd -
fi

eval "$(conda shell.bash hook)"
conda activate cosmos-predict2.5
source .venv/bin/activate

unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE

torchrun --nproc_per_node=8 --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  -- experiment=robotwin_14d_8gpu \
  job.name=robotwin_14d_v01_16f_224_8gpu \
  trainer.max_iter=100000 \
  model.config.state_t=5 \
  model.config.net.num_action_per_chunk=16 \
  dataloader_train.sampler.dataset.num_action_per_chunk=16 \
  dataloader_train.sampler.dataset.fps_downsample_ratio=1 \
  'dataloader_train.sampler.dataset.video_size=[224,224]' \
  dataloader_train.dataset.num_action_per_chunk=16 \
  dataloader_train.dataset.fps_downsample_ratio=1 \
  'dataloader_train.dataset.video_size=[224,224]' \
  ~dataloader_train.dataloaders
