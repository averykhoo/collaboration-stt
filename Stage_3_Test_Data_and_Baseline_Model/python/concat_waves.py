#!/usr/bin/env python3
import sys
from pathlib import Path
import soundfile as sf
import numpy as np

def main():
    if len(sys.argv) != 3:
        print("Usage: concat_waves.py <in_dir> <out.wav>")
        sys.exit(1)

    in_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    # Collect all .wav files, sort by numeric filename if possible
    wav_files = sorted(
        in_dir.glob("*.wav"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
    )

    if not wav_files:
        print(f"No .wav files found in {in_dir}")
        sys.exit(1)

    data_list = []
    samplerate = None

    for wav_file in wav_files:
        data, sr = sf.read(wav_file)
        if samplerate is None:
            samplerate = sr
        elif sr != samplerate:
            print(f"Warning: {wav_file} has a different samplerate ({sr}), resampling skipped.")
        data_list.append(data)

    # Concatenate all
    concat_data = np.concatenate(data_list)

    # Write output
    sf.write(out_path, concat_data, samplerate)
    print(f"✅ Wrote {len(wav_files)} files to {out_path}")

if __name__ == "__main__":
    main()
