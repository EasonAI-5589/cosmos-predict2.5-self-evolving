#!/bin/bash

# 设置代理（允许下载）
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000

# 取消 HuggingFace 离线模式，允许使用本地缓存
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE

# Image2World 推理
python examples/inference.py \
  -i assets/base/robot_welding.json \
  -o outputs/test_image2world \
  --inference-type=image2world \

echo "推理完成，输出目录: outputs/test_image2world"
