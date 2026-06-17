#!/bin/bash

export PYTHONPATH='/mount/data/alx/Stage_4_Train/k2_2025/icefall/egs/tedlium3/ASR/zipformer:/mount/data/alx/Stage_4_Train/k2_2025/icefall:$PYTHONPATH'

export CUDA_VISIBLE_DEVICES="0"
epoch=50
avg=20

# create average model
python ./zipformer/export-onnx.py \
  --tokens data/lang_bpe_500/tokens.txt \
  --use-averaged-model 1 \
  --epoch ${epoch} \
  --avg ${avg} \
  --exp-dir zipformer/exp



  