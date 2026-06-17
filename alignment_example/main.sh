#!/bin/bash

# NOTE: Install dependencies first using: pip install -r requirements.txt

stage=0


# At this stage we setup inter directory and convert PDF to text
if [ $stage -le 0 ]; then
    echo stage 0
    rm -f -r inter
    mkdir inter
    pip install pdfminer.six
    python python/pdf2txt.py input/DR-14102024.pdf inter/test.txt
fi

# At this stage we convert OPUS to WAV
if [ $stage -le 1 ]; then
    echo stage 1
    ffmpeg -y -i input/20241014.16k.mono.opus \
    -ac 1 \
    -ar 16000 \
    -sample_fmt s16 \
    -c:a pcm_s16le \
    inter/test.wav
fi

# At this stage we apply VAD (voice activity detection) to WAV file
if [ $stage -le 2 ]; then
    echo stage 2
    python python/apply_vad.py inter/test.wav inter/test_vad.txt
fi

# At this stage we transcribe WAV file using Whisper.
if [ $stage -le 3 ]; then
    echo stage 3
    python python/whisper_transcribe.py inter/test.wav inter/test_vad.txt inter/test_whisper_out.txt
fi

# Normalize Whisper out to json format
if [ $stage -le 4 ]; then
    echo stage 4
    python python/norm_whisper_out.py inter/test_whisper_out.txt inter/test_whisper_out.json
fi

# Parse PDF copy
if [ $stage -le 5 ]; then
    echo stage 5
    python python/parse_pdf_copy.py inter/test.txt inter/gt.json
fi

# PDF json 2 txt
if [ $stage -le 6 ]; then
    echo stage 6
    python python/gt_json_to_txt.py inter/gt.json inter/gt.txt
fi

# Align Whisper output with GT text from PDF
if [ $stage -le 7 ]; then
    echo stage 7
    python python/dtw_json_txt.py inter/gt.txt inter/test_whisper_out.json inter/test_whisper_out_ali.json
fi


# It is recommened to use segments from inter/test_whisper_out_ali.json for ASR trainng.
# But for training we recommend to use only segments with CER < 20%.

