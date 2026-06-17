#!/bin/bash

# Total execution time: ~1 hour.

# =============================================================================
# Stage 3: Evaluate test data using Whisper model
# =============================================================================
# This script evaluates Malay speech recognition using Whisper on 3 benchmarks:
#   1. Fleurs (clean)     - Concatenated test utterances from Fleurs dataset
#   2. Rerecorded (noisy) - 30 min played via Mac, recorded via iPhone WhatsApp at 3m
#   3. Mesolitica         - Mesolitica test file with STM ground truth
#
# Pipeline: Download -> Concatenate -> VAD -> Whisper -> Normalize -> WER
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration Variables
# -----------------------------------------------------------------------------
# Starting stage number (0-8). Set to higher number to skip earlier stages.
stage=0

# Directory for intermediate files (VAD output, transcriptions, normalized text)
base_path=./inter

# Enable/disable parallel Whisper transcription.
# Set to 'false' if you encounter CUDA Out-of-Memory (OOM) errors.
# When false, transcription runs sequentially (one file at a time).
enable_parallel_transcribe=false

# Path to rerecorded noisy audio (30 min of Fleurs played and re-recorded at 3m distance)
rerecorded_wav_path=./input/fleurs_test_30min_noisy.wav

# Ground truth text for the rerecorded audio
rerecorded_gt=./input/fleurs_all_test_30min.txt

# Mesolitica test audio file path
mesolitica_test_wav_path=./input/malay_000001.wav

# Mesolitica ground truth in STM format (will be converted to plain text)
mesolitica_test_stm_path=./input/malay_000001.stm

# =============================================================================
# Stage 0: Environment Setup
# =============================================================================
# Install Python dependencies and create directory structure for test data.
# This stage should only run once during initial setup.
if [ $stage -le 0 ]; then
    echo stage 0
    pip install -r requirements.txt --break-system-packages
    mkdir -p ${base_path}/test
    python python/download_fleurs_test.py ${base_path}/test
fi

# =============================================================================
# Stage 1: Create Concatenated Test Audio (Fleurs)
# =============================================================================
# Concatenate all individual Fleurs test utterances into a single WAV file.
# This creates one continuous audio file for batch evaluation.
# Input:  ${base_path}/test/ (individual WAV files)
# Output: ${base_path}/fleurs.wav (concatenated audio)
if [ $stage -le 1 ]; then
    echo stage 1
    python python/concat_waves.py ${base_path}/test ${base_path}/fleurs.wav
fi

# =============================================================================
# Stage 2: Create Concatenated Ground Truth Text (Fleurs)
# =============================================================================
# Concatenate all individual Fleurs transcripts into a single text file.
# This creates the reference text for WER calculation.
# Input:  ${base_path}/test/ (individual TXT files)
# Output: ${base_path}/fleurs.txt (concatenated ground truth)
if [ $stage -le 2 ]; then
    echo stage 2
    python python/concat_txts.py ${base_path}/test ${base_path}/fleurs.txt
fi

# =============================================================================
# Stage 3: Prepare Mesolitica Benchmark Data
# =============================================================================
# Copy Mesolitica test files and convert STM ground truth to plain text.
# STM (Speech Transcription Markup) format contains timestamps and speaker info
# which are stripped to create raw text for WER evaluation.
# Input:  ${mesolitica_test_wav_path}, ${mesolitica_test_stm_path}
# Output: ${base_path}/mesolitica.wav, ${base_path}/mesolitica.txt
if [ $stage -le 3 ]; then
    echo stage 3
    cp ${mesolitica_test_wav_path} ${base_path}/mesolitica.wav
    cp ${mesolitica_test_stm_path} ${base_path}/mesolitica.stm
    python python/stm2txt.py ${base_path}/mesolitica.stm ${base_path}/mesolitica.txt
fi

# =============================================================================
# Stage 4: Prepare Rerecorded (Noisy) Benchmark Data
# =============================================================================
# Copy rerecorded audio and its ground truth to intermediate folder.
# This benchmark simulates distant microphone recording conditions:
# - Original: First 30 min of Fleurs concatenated audio
# - Recording setup: Mac playback -> 3 meter distance -> iPhone WhatsApp recording
# Input:  ${rerecorded_wav_path}, ${rerecorded_gt}
# Output: ${base_path}/rerecorded.wav, ${base_path}/rerecorded.txt
if [ $stage -le 4 ]; then
    echo stage 4
    cp ${rerecorded_wav_path} ${base_path}/rerecorded.wav
    cp ${rerecorded_gt} ${base_path}/rerecorded.txt
fi

# =============================================================================
# Stage 5: Whisper Transcription
# =============================================================================
# Transcribe audio using OpenAI Whisper model guided by VAD segments.
# Parallel mode runs all 3 transcriptions simultaneously (requires more GPU memory).
# Sequential mode runs one at a time (use if CUDA OOM occurs).
# Input:  ${base_path}/*.wav (audio), ${base_path}/*_vad.txt (VAD segments)
# Output: ${base_path}/*_whisper_out.txt (raw Whisper transcription output)
if [ $stage -le 6 ]; then
    echo stage 6
    if [ "$enable_parallel_transcribe" = true ]; then
        echo "Running Whisper transcription in parallel mode"
        python python/whisper_transcribe.py ${base_path}/fleurs.wav ${base_path}/fleurs_whisper_out.txt &
        python python/whisper_transcribe.py ${base_path}/mesolitica.wav  ${base_path}/mesolitica_whisper_out.txt &
        python python/whisper_transcribe.py ${base_path}/rerecorded.wav  ${base_path}/rerecorded_whisper_out.txt &
        wait
    else
        echo "Running Whisper transcription in sequential mode (parallel disabled)"
        python python/whisper_transcribe.py ${base_path}/fleurs.wav  ${base_path}/fleurs_whisper_out.txt
        python python/whisper_transcribe.py ${base_path}/mesolitica.wav  ${base_path}/mesolitica_whisper_out.txt
        python python/whisper_transcribe.py ${base_path}/rerecorded.wav  ${base_path}/rerecorded_whisper_out.txt
    fi
fi

# =============================================================================
# Stage 7: Text Extraction and Normalization
# =============================================================================
# Two-step process:
# 1. Extract plain text from Whisper JSON/structured output
# 2. Normalize both Whisper output and ground truth for fair WER comparison
#
# Normalization includes:
#   - Lowercase conversion
#   - Remove thousand separators (30,000 -> 30000)
#   - Expand dates (dd/mm/yyyy -> Malay words)
#   - Expand number ranges (10-15 -> sepuluh hingga lima belas)
#   - Expand currency (RM150 -> seratus lima puluh ringgit)
#   - Expand units (5km -> lima kilometer)
#   - Expand decimals and percentages
#   - Remove punctuation
#   - Convert all integers to Malay words
#   - Collapse whitespace
#
# Input:  ${base_path}/*_whisper_out.txt, ${base_path}/*.txt (ground truth)
# Output: ${base_path}/*_whisper_out_text_norm.txt, ${base_path}/*_norm.txt
if [ $stage -le 7 ]; then
    echo stage 7
    # Step 1: Extract plain text from Whisper output
    python python/extract_whisper_text.py ${base_path}/fleurs_whisper_out.txt ${base_path}/fleurs_whisper_out_text.txt
    python python/extract_whisper_text.py ${base_path}/mesolitica_whisper_out.txt ${base_path}/mesolitica_whisper_out_text.txt
    python python/extract_whisper_text.py ${base_path}/rerecorded_whisper_out.txt ${base_path}/rerecorded_whisper_out_text.txt
    
    # Step 2: Normalize Whisper transcription output
    python python/norm_malay_text.py ${base_path}/fleurs_whisper_out_text.txt ${base_path}/fleurs_whisper_out_text_norm.txt
    python python/norm_malay_text.py ${base_path}/mesolitica_whisper_out_text.txt ${base_path}/mesolitica_whisper_out_text_norm.txt
    python python/norm_malay_text.py ${base_path}/rerecorded_whisper_out_text.txt ${base_path}/rerecorded_whisper_out_text_norm.txt
    
    # Step 3: Normalize ground truth reference text (same normalization as hypothesis)
    python python/norm_malay_text.py ${base_path}/fleurs.txt ${base_path}/fleurs_norm.txt
    python python/norm_malay_text.py ${base_path}/mesolitica.txt ${base_path}/mesolitica_norm.txt
    python python/norm_malay_text.py ${base_path}/rerecorded.txt ${base_path}/rerecorded_norm.txt
fi

# =============================================================================
# Stage 8: Word Error Rate (WER) Calculation
# =============================================================================
# Calculate WER for each benchmark comparing normalized hypothesis vs reference.
# WER = (Substitutions + Insertions + Deletions) / Total Reference Words
#
# The calculation uses allowed lists to handle acceptable variations:
#   - allowed_replacements.txt: Word pairs that should not count as errors
#   - allowed_insertions.txt: Words that can be inserted without penalty
#   - allowed_deletions.txt: Words that can be deleted without penalty
#
# Runs in parallel for all 3 benchmarks.
# Input:  ${base_path}/*_norm.txt (reference), ${base_path}/*_whisper_out_text_norm.txt (hypothesis)
# Output: ${base_path}/*_stat.txt (WER statistics including S/I/D breakdown)
if [ $stage -le 8 ]; then
    echo stage 8
    python python/wer.py ${base_path}/fleurs_norm.txt ${base_path}/fleurs_whisper_out_text_norm.txt allowed_replacements.txt allowed_insertions.txt allowed_deletions.txt ${base_path}/fleurs_stat.txt &
    python python/wer.py ${base_path}/mesolitica_norm.txt ${base_path}/mesolitica_whisper_out_text_norm.txt allowed_replacements.txt allowed_insertions.txt allowed_deletions.txt ${base_path}/mesolitica_stat.txt &
    python python/wer.py ${base_path}/rerecorded_norm.txt ${base_path}/rerecorded_whisper_out_text_norm.txt allowed_replacements.txt allowed_insertions.txt allowed_deletions.txt ${base_path}/rerecorded_stat.txt &
    wait
fi

