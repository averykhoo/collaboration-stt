#!/usr/bin/env bash

# Total execution time: ~1 hour.

stage=0
lang=ms_my
musan_dir=/mount/data/alx/intermidiate/musan/musan
inter_dir=/mount/data/alx/intermidiate
icefall_dir=/mount/data/alx/Stage_4_Train/k2_2025/icefall

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export PYTHONPATH=${icefall_dir}:$PYTHONPATH

set -eou pipefail


vocab_sizes=(
  500
)

# Create test/dev/train files
if [ $stage -le 0 ]; then
  echo 0: Create test/dev/train files
  python python/create_test_dev_train.py ${inter_dir}/stm ${inter_dir}
fi

# Create musan manifests
if [ $stage -le 1 ]; then
  echo 1: Prepare musan manifests
  lhotse prepare musan ${musan_dir} data/manifests
fi

# Create dev manifests
if [ $stage -le 2 ]; then
  echo stage 2 - create dev manifests
  mkdir -p data/manifests
  python python/create_k2_manifest_16k.py \
                            ${inter_dir}/dev.list \
                            tedlium \
                            dev \
                            ${inter_dir}/wav \
                            ${inter_dir}/stm \
                            data/manifests \
                            ${lang}
  gzip data/manifests/tedlium_recordings_dev.jsonl
  gzip data/manifests/tedlium_supervisions_dev.jsonl
fi

# Create test manifests
if [ $stage -le 3 ]; then
  echo stage 3 - create test manifests
  python python/create_k2_manifest_16k.py \
                            ${inter_dir}/test.list \
                            tedlium \
                            test \
                            ${inter_dir}/wav \
                            ${inter_dir}/stm \
                            data/manifests \
                            ${lang}
  gzip data/manifests/tedlium_recordings_test.jsonl
  gzip data/manifests/tedlium_supervisions_test.jsonl
fi

# Create train manifests
if [ $stage -le 4 ]; then
  echo stage 4 - create train manifests
  date
  python python/create_k2_manifest_16k.py \
                            ${inter_dir}/train.list \
                            tedlium \
                            train \
                            ${inter_dir}/wav \
                            ${inter_dir}/stm \
                            data/manifests \
                            ${lang}
  gzip data/manifests/tedlium_recordings_train.jsonl
  gzip data/manifests/tedlium_supervisions_train.jsonl
  date
  echo stage 4 - end
fi

# Compute fbank for tedlium3 and musan
if [ $stage -le 5 ] ; then
  echo "Stage 5: Compute fbank for tedlium3 and musan"
  rm -fr data/fbank
  mkdir data/fbank
  echo fbank tedlium
  date  
  python3 local/compute_fbank_tedlium_1.py
  echo fbank musan - start
  date
  python3 local/compute_fbank_musan.py
  echo fbank musan - end
  date
fi

if [ $stage -le 6 ] ; then
  echo "Stage 6: Prepare BPE train data and set of words"
  lang_dir=data/lang
  mkdir -p $lang_dir

  cp -f ${inter_dir}/trans_all.txt $lang_dir/train.txt
  cp -f ${inter_dir}/words_orig.txt $lang_dir/words_orig.txt
  cp -f ${inter_dir}/words.txt $lang_dir/words.txt  

fi

if [ $stage -le 7 ]; then
  echo "Stage 7: Prepare BPE based lang"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    mkdir -p $lang_dir

    cp data/lang/words.txt $lang_dir    

    ./local/train_bpe_model.py \
      --lang-dir $lang_dir \
      --vocab-size $vocab_size \
      --transcript data/lang/train.txt

    if [ ! -f $lang_dir/L_disambig.pt ]; then
     ./local/prepare_lang_bpe.py --lang-dir $lang_dir --oov "<unk>"
    fi
  done
fi

echo " " > prepare.done
