#!/usr/bin/env python3

import sys
import whisper
import torch
import torchaudio


def read_vad_file(path):
    """
    Reads VAD file with format:
    start_sec end_sec
    """
    segments = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                continue

            start, end = map(float, parts)
            if end > start:
                segments.append((start, end))

    return segments


def load_audio_16k(path):
    """
    Load audio and resample to 16kHz mono using torchaudio
    Returns 1D torch.Tensor
    """
    wav, sr = torchaudio.load(path)  # shape: [channels, samples]

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    return wav.squeeze(0)  # [samples]


def main():
    if len(sys.argv) != 4:
        print(
            "usage: python whisper_transcribe.py "
            "in_wav vad_txt out_txt"
        )
        sys.exit(1)

    in_wav = sys.argv[1]
    vad_txt = sys.argv[2]
    out_txt = sys.argv[3]

    # Load Whisper large-v3
    whisper_model = whisper.load_model("large-v3")

    # Load audio (16 kHz, mono)
    wav = load_audio_16k(in_wav)

    # Read external VAD segments
    vad_segments = read_vad_file(vad_txt)

    with open(out_txt, "w", encoding="utf-8") as fout:
        for start_sec, end_sec in vad_segments:
            start_sample = int(start_sec * 16000)
            end_sample = int(end_sec * 16000)

            audio_chunk = wav[start_sample:end_sample]
            if audio_chunk.numel() == 0:
                continue

            audio_chunk = whisper.pad_or_trim(audio_chunk)

            # IMPORTANT: large-v3 requires 128 mel bins
            mel = whisper.log_mel_spectrogram(
                audio_chunk,
                n_mels=whisper_model.dims.n_mels
            ).to(whisper_model.device)

            result = whisper.decode(
                whisper_model,
                mel,
                whisper.DecodingOptions(
                    language="ms",
                    task="transcribe",
                    beam_size=5
                )
            )

            text = result.text.strip()
            if not text:
                continue

            fout.write(
                f"[{start_sec:.2f} {end_sec:.2f}] {text}\n"
            )


if __name__ == "__main__":
    main()
