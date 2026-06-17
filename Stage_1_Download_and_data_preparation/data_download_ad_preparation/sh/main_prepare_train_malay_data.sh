#!/bin/bash

# Total execution time: ~15 hours.

# Pipeline stage control:
# Set `stage` to skip earlier stages and start from a later one
stage=0

# Intermediate working directory
inter_path=/mount/data/alx/intermidiate/

# Save current working directory so we can return to it later
cur_dir=`pwd`

# =========================
# STAGE 0: Extract audio and text
# =========================
# This stage installs required Python dependencies and runs a Python script
# that extracts WAV audio files and text transcriptions from parquet files.
# The output is typically organized into shard-based directories.
if [ $stage -le 0 ]; then
    echo stage 0

    # Move back to the base working directory
    cd ${inter_path}

    # Install Python dependencies needed for extraction
    pip install datasets soundfile pyarrow librosa

    # Run the extraction script
    python ${cur_dir}/python/extract_waves_texts.py

    # Return to the original working directory
    cd ${cur_dir}
fi

# ------------------------------------------------------------------
# Stage 1:
# - Generate STM and WAV training data from extracted input.
# ------------------------------------------------------------------
if [ $stage -le 1 ]; then
    echo stage 1
    rm -fr ${inter_path}/stm ${inter_path}/wav
    mkdir ${inter_path}/stm ${inter_path}/wav    
    python python/create_train_data.py ${inter_path}/extracted  ${inter_path}/stm ${inter_path}/wav
fi

# ------------------------------------------------------------------
# Stage 2:
# - Compute letter statistics from STM files
# ------------------------------------------------------------------
if [ $stage -le 2 ]; then
    echo stage 2
    python python/create_letter_stat.py ${inter_path}/stm ${inter_path}/letter.stat
fi

# ------------------------------------------------------------------
# Stage 3:
# - Create a single combined transcription file (trans_all.txt)
# - Aggregates all STM transcripts into one text file
# ------------------------------------------------------------------
if [ $stage -le 3 ]; then
    echo stage 3
    python python/create_trans_all.py ${inter_path}/stm ${inter_path}/trans_all.txt
fi

# ------------------------------------------------------------------
# Stage 4:
# - Generate original word list from the combined transcription
# ------------------------------------------------------------------
if [ $stage -le 4 ]; then
    echo stage 4
    python python/create_word_list.py ${inter_path}/trans_all.txt ${inter_path}/words_orig.txt
fi

# ------------------------------------------------------------------
# Stage 5:
# - For every word define int ID
# ------------------------------------------------------------------
if [ $stage -le 5 ]; then
    echo stage 5
    python python/create_words_2_id.py ${inter_path}/words_orig.txt ${inter_path}/words.txt
fi

# ------------------------------------------------------------------
# Stage 6:
# - Calculate total net audio time (in hours) from STM files
# - Print the resulting net time to stdout
# ------------------------------------------------------------------
if [ $stage -le 6 ]; then
    echo stage 6
    python python/calculate_net_time_from_stms.py ${inter_path}/stm ${inter_path}/net_time.txt
    cat ${inter_path}/net_time.txt
fi
