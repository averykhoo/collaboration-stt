#!/bin/bash

# Execution time: ~7 days.

wdir=/mount/data/alx/Stage_4_Train/k2_2025/icefall/egs/tedlium3/ASR
num_gpus=1

export PYTHONPATH='/mount/data/alx/Stage_4_Train/k2_2025/icefall:$PYTHONPATH'

nvidia-smi

export CUDA_VISIBLE_DEVICES="0"
PYTHONWARNINGS='ignore::FutureWarning' python ./zipformer/train.py \
  --use-fp16 true \
  --world-size ${num_gpus} \
  --num-epochs 50 \
  --start-epoch 1 \
  --exp-dir zipformer/exp \
  --max-duration 1000 # 3000 for h200
