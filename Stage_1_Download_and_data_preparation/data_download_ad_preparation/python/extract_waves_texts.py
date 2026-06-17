#!/usr/bin/env python3
# Shebang to allow running the script directly with Python 3

from pathlib import Path
import pyarrow.parquet as pq
import soundfile as sf
import numpy as np
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import librosa


# =========================
# CONFIGURATION
# =========================

# Root directory of the dataset
DATASET_ROOT = Path("malaya-speech-malay-stt")

# Directory containing parquet files
PARQUET_DIR = DATASET_ROOT / "data"

# Output root directory where extracted data will be stored
OUT_ROOT = Path("extracted")
OUT_ROOT.mkdir(exist_ok=True)

# Number of samples per shard (used to split output into folders)
SHARD_SIZE = 100_000

# Filename format for extracted files (zero-padded index)
FILENAME_FMT = "{idx:07d}"

# Target sampling rate for audio output
TARGET_SR = 16_000

# Maximum number of parallel processes
MAX_PROCESSES = 64


# =========================
# HELPER FUNCTION
# =========================

def process_parquet(args):
    """
    Processes ONE parquet file.
    Returns the number of processed samples.

    This function is executed in a separate process.
    """
    # Unpack arguments: parquet file path and starting global index
    pq_file, start_idx = args

    # Counter for samples processed in this parquet file
    local_count = 0

    # Read the entire parquet file into a PyArrow table
    table = pq.read_table(pq_file)

    # Convert the table into a list of Python dictionaries (one per row)
    rows = table.to_pylist()

    # Iterate over all rows in the parquet file
    for i, row in enumerate(rows):
        # Global index of the sample across all parquet files
        global_idx = start_idx + i

        # Determine shard ID based on global index
        shard_id = global_idx // SHARD_SIZE

        # Create shard-specific directories for audio and text
        wav_dir = OUT_ROOT / f"waves_{shard_id}"
        txt_dir = OUT_ROOT / f"texts_{shard_id}"
        wav_dir.mkdir(parents=True, exist_ok=True)
        txt_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename using global index
        name = FILENAME_FMT.format(idx=global_idx)

        # Full paths for output audio and text files
        wav_path = wav_dir / f"{name}.wav"
        txt_path = txt_dir / f"{name}.txt"

        # ---------- AUDIO ----------
        # Extract raw audio bytes from the parquet row
        audio_bytes = row["filename"]["bytes"]

        # Read audio from memory buffer into float32 numpy array
        audio, sr = sf.read(BytesIO(audio_bytes), dtype="float32")

        # If audio is stereo, convert to mono by averaging channels
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Resample audio if sampling rate does not match target
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        # Write audio to WAV file using 16-bit PCM encoding
        sf.write(
            wav_path,
            audio,
            TARGET_SR,
            subtype="PCM_16"
        )

        # ---------- TEXT ----------
        # Write transcription text to file (strip removes extra whitespace)
        txt_path.write_text(row["Y"].strip(), encoding="utf-8")

        # Increment local processed sample counter
        local_count += 1

    # Return number of samples processed in this parquet file
    return local_count


# =========================
# MAIN
# =========================

def main():
    # Collect and sort all parquet files in the dataset directory
    parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))

    # Ensure that parquet files exist
    assert parquet_files, "No parquet files found"

    print(f"Found {len(parquet_files)} parquet files")

    # Compute starting global index for each parquet file
    # This ensures unique filenames across all files
    starts = []
    current = 0

    for pq_file in parquet_files:
        # Read only the "Y" column to count rows efficiently
        table = pq.read_table(pq_file, columns=["Y"])
        n = table.num_rows

        # Store starting index for this parquet file
        starts.append(current)

        # Update global index offset
        current += n

    # Total number of samples across all parquet files
    total_samples = current
    print(f"Total samples: {total_samples}")

    # Pair each parquet file with its starting index
    tasks = list(zip(parquet_files, starts))

    # Limit number of worker processes
    workers = min(MAX_PROCESSES, len(tasks))
    print(f"Using {workers} processes")

    # Counter for all processed samples
    processed = 0

    # Create a process pool for parallel processing
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all parquet processing tasks
        futures = [executor.submit(process_parquet, t) for t in tasks]

        # Iterate over completed tasks with progress bar
        for f in tqdm(as_completed(futures), total=len(futures), desc="Parquet files"):
            processed += f.result()

    print("Done.")
    print(f"Total extracted samples: {processed}")


# Entry point of the script
if __name__ == "__main__":
    main()
