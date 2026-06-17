#!/usr/bin/env python3
# create_k2_data.py
#
# This script prepares data for K2 / Kaldi-style training.
# It takes extracted WAV + TXT pairs and packs them into:
#   - long WAV files (up to MAX_DURATION seconds)
#   - corresponding STM files with segment timing
#
# Usage:
#   python create_k2_data.py <in_dataset_dir> <out_stm_dir> <out_wav_dir>

import os
import sys
import re
import numpy as np
from pathlib import Path
import soundfile as sf

# Try to import SciPy-based resampling.
# If unavailable, the script will fall back to a NumPy-only method.
try:
    from scipy.signal import resample_poly
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Target sample rate for all output audio
TARGET_SR = 16000

# Maximum duration (seconds) of one packed WAV file
# Typical value for K2 / Kaldi-style long recordings
MAX_DURATION = 600.0

# Extra tolerance to avoid excessive flushing due to rounding errors
TOLERANCE = 2.0

# Prefix for output file IDs
OUT_PREFIX = "malay_"

def _natural_key(s):
    """
    Generate a sorting key that orders strings naturally.
    Example:
        file2 < file10 (instead of lexicographic order)
    """
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def list_pairwise_text_wav(in_root: Path):
    """
    Scan the input directory and find matching TXT and WAV file pairs.

    Expected structure:
        texts_<N>/*.txt
        waves_<N>/*.wav

    Only files with matching base names and shard indices are paired.
    """
    text_dirs, wav_dirs = [], []

    # Collect text and wave shard directories
    for p in sorted(in_root.iterdir()):
        if p.is_dir() and p.name.startswith("texts_"):
            text_dirs.append(p)
        elif p.is_dir() and p.name.startswith("waves_"):
            wav_dirs.append(p)

    # Extract numeric shard index from directory name
    def idx(name): return int(re.search(r"_(\d+)$", name).group(1))

    # Map shard index → directory
    text_map = {idx(d.name): d for d in text_dirs}
    wav_map = {idx(d.name): d for d in wav_dirs}

    pairs = []

    # Iterate over shard indices that exist in both text and wave directories
    for i in sorted(set(text_map) & set(wav_map)):
        tdir, wdir = text_map[i], wav_map[i]

        # Map basename → full path
        t_files = {p.stem: p for p in tdir.glob("*.txt")}
        w_files = {p.stem: p for p in wdir.glob("*.wav")}

        # Pair TXT and WAV files with the same stem
        for key in sorted(set(t_files) & set(w_files), key=_natural_key):
            pairs.append((t_files[key], w_files[key]))

    return pairs

def load_audio(path: Path, target_sr: int):
    """
    Load audio from disk, convert to mono, and resample to target_sr.

    Uses:
    - scipy.signal.resample_poly if SciPy is available (higher quality)
    - NumPy interpolation fallback otherwise
    """
    data, sr = sf.read(str(path))

    # Convert stereo to mono if needed
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Resample if sampling rate differs
    if sr != target_sr:
        if _HAS_SCIPY:
            # Rational resampling using polyphase filtering
            from math import gcd
            g = gcd(sr, target_sr)
            data = resample_poly(data, target_sr // g, sr // g)
        else:
            # Simple linear interpolation fallback
            duration = len(data) / sr
            n_out = int(duration * target_sr)
            data = np.interp(
                np.linspace(0, 1, n_out, endpoint=False),
                np.linspace(0, 1, len(data), endpoint=False),
                data,
            )

    return data.astype(np.float32)

def write_wav(path, data, sr):
    """
    Write audio data to a WAV file using 16-bit PCM encoding.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16")

def write_stm(path, lines):
    """
    Write STM transcription file.

    Each line corresponds to one speech segment with timing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def format_stm_line(fid, start, end, text):
    """
    Format a single STM line according to Kaldi/K2 conventions.
    """
    return f"{fid} 1 {fid} {start:.2f} {end:.2f} <NA> {text.strip()}"

def main():
    # Validate command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python create_k2_data.py <in_dataset_dir> <out_stm_dir> <out_wav_dir>")
        sys.exit(1)

    # Input directory containing texts_*/waves_* directories
    in_root = Path(sys.argv[1])

    # Output directories for STM and WAV files
    out_stm_dir = Path(sys.argv[2])
    out_wav_dir = Path(sys.argv[3])

    # Collect matching (TXT, WAV) pairs
    pairs = list_pairwise_text_wav(in_root)

    # Debug print of one example pair (sanity check)
    print(str(pairs[10000]))

    if not pairs:
        print("No matching pairs found.")
        sys.exit(1)

    # Index of the current packed output file
    pack_idx = 1

    # Accumulators for current pack
    cur_audio, cur_stm, cur_time = [], [], 0.0

    def flush():
        """
        Finalize the current pack:
        - Concatenate audio
        - Write WAV and STM files
        - Reset accumulators
        """
        nonlocal pack_idx, cur_audio, cur_stm, cur_time

        if not cur_audio:
            return

        fid = f"{OUT_PREFIX}{pack_idx:06d}"
        wav_path = out_wav_dir / f"{fid}.wav"
        stm_path = out_stm_dir / f"{fid}.stm"

        audio = np.concatenate(cur_audio)

        write_wav(wav_path, audio, TARGET_SR)
        write_stm(stm_path, cur_stm)

        print(f"Wrote {wav_path} ({len(audio)/TARGET_SR:.2f}s), {stm_path} ({len(cur_stm)} segs)")

        pack_idx += 1
        cur_audio, cur_stm, cur_time = [], [], 0.0

    # Iterate over all text/audio pairs
    for tpath, wpath in pairs:
        # Load text and audio
        txt = tpath.read_text(encoding="utf-8", errors="ignore").strip()
        aud = load_audio(wpath, TARGET_SR)

        dur = len(aud) / TARGET_SR
        projected = cur_time + dur

        # If adding this segment would exceed max duration, flush first
        if projected > MAX_DURATION + TOLERANCE:
            flush()

        fid = f"{OUT_PREFIX}{pack_idx:06d}"

        # Add STM entry for this segment
        cur_stm.append(format_stm_line(fid, cur_time, cur_time + dur, txt))

        # Append audio and update time cursor
        cur_audio.append(aud)
        cur_time += dur

        # Flush again if we crossed the limit after adding
        if cur_time >= MAX_DURATION + TOLERANCE:
            flush()

    # Flush any remaining data
    flush()

if __name__ == "__main__":
    main()
