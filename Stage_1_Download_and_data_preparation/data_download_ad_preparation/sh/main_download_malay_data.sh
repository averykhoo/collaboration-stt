#!/bin/bash
# Use Bash shell to execute this script

# Total execution time: ~1 hour.

# In this script we show how to download the Malay database for STT training.
# The dataset is hosted on Hugging Face and stored using Git LFS.
# Please note that all textual data in this database is already normalized (numbers/lower-upper case etc..),
# therefore no additional text normalization stage is required. 
# Note, that additional preparation is needed and will be executed in the attached python notebook.

# Stage control variable:
# Allows running the script step-by-step by increasing the stage value.
# For example:
#   stage=0 → run all stages (stage 0 - git clone + stage 1 - download)
#   stage=1 → skip stage 0, run stage 1 and later
stage=0

# Base path where the dataset will be downloaded and processed
db_path=/mount/data/alx/intermidiate

# Save current working directory so we can return to it later
cur_dir=`pwd`

# =========================
# STAGE 0: Git clone (pointers only)
# =========================
# This stage clones the dataset repository from Hugging Face.
# GIT_LFS_SKIP_SMUDGE=1 prevents Git LFS from downloading large files.
# As a result, only lightweight pointer files (.parquet references) are downloaded.
if [ $stage -le 0 ]; then
    echo stage 0
    cd ${db_path}
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/mesolitica/malaya-speech-malay-stt
fi

# =========================
# STAGE 1: Git LFS download (real parquet files)
# =========================
# This stage installs Git LFS and downloads the actual large parquet files.
# Git LFS replaces the previously downloaded pointer files with real data.
if [ $stage -le 1 ]; then
    echo stage 1

    git config --global lfs.activitytimeout 600
    git config --global lfs.transfer.maxretries 10

    # Update package lists
    apt update

    # Install Git LFS (required to download large files)
    apt install -y git-lfs

    # Initialize Git LFS in the environment
    git lfs install

    # Mark the dataset directory as safe (required in some container environments)
    git config --global --add safe.directory ${db_path}/malaya-speech-malay-stt

    # Move into the dataset repository
    cd ${db_path}/malaya-speech-malay-stt

    # Download all large files managed by Git LFS (parquet files)
    git lfs pull
fi

