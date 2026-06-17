#!/usr/bin/env python3

import sys
import whisper
import torch

def main():
    # Only requires input wav and output txt paths
    if len(sys.argv) != 3:
        print(
            "usage: python whisper_transcribe.py "
            "in_wav out_txt"
        )
        sys.exit(1)

    in_wav = sys.argv[1]
    out_txt = sys.argv[2]

    # Load Whisper large-v3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model("large-v3", device=device)

    # .transcribe() handles resampling, VAD, and windowing automatically
    result = whisper_model.transcribe(
        in_wav,
        language="ms",
        task="transcribe",
        beam_size=5,
        verbose=False
    )

    with open(out_txt, "w", encoding="utf-8") as fout:
        # Whisper returns a list of segments with timestamps
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            
            if text:
                fout.write(f"[{start:.2f} {end:.2f}] {text}\n")

if __name__ == "__main__":
    main()