#!/usr/bin/env python3
"""
create_letter_stat.py <stm_dir> <output_stat_file>

This script scans a directory tree for .stm files and computes
letter frequency statistics.

Key points:
- Works recursively over all subdirectories
- Counts ONLY alphabetic characters (Unicode-aware)
- Ignores punctuation, numbers, spaces, and symbols
- Reads transcription text starting from the 7th token of each STM line
- Uses multiprocessing to speed up processing on large corpora
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import unicodedata
from typing import Iterable, Tuple

# Maximum number of worker processes allowed
MAX_PROCESSES = 400

def iter_stm_files(root: Path) -> Iterable[Path]:
    """
    Recursively iterate over a directory tree and yield all .stm files.
    """
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".stm"):
                yield Path(dirpath) / fn

def count_letters_in_line(line: str) -> Counter:
    """
    Count alphabetic Unicode characters in a single STM line.

    Rules:
    - Skip empty lines
    - Skip comment lines (starting with ';')
    - Require at least 7 whitespace-separated fields
    - Normalize Unicode text to NFC
    - Convert all letters to lowercase before counting
    """
    line = line.strip()

    # Ignore empty lines and STM comments
    if not line or line.startswith(";"):
        return Counter()

    parts = line.split()

    # STM format requires at least 7 fields before transcription
    if len(parts) < 7:
        return Counter()

    # Transcription text begins at token index 6
    transcript = " ".join(parts[6:])

    # Normalize Unicode to canonical composed form
    transcript = unicodedata.normalize("NFC", transcript)

    c = Counter()

    # Count only alphabetic Unicode characters
    for ch in transcript:
        if ch.isalpha():
            c[ch.lower()] += 1

    return c

def process_stm_file(path: Path) -> Tuple[str, Counter]:
    """
    Process a single STM file and return letter counts.

    Returns:
        (path_as_string, Counter of letter frequencies)

    Errors are silently ignored to allow robust batch processing.
    """
    counts = Counter()
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                counts.update(count_letters_in_line(line))
    except Exception:
        # On error, return empty counts for this file
        return (str(path), counts)

    return (str(path), counts)

def merge_counters(counters: Iterable[Counter]) -> Counter:
    """
    Merge multiple Counter objects into a single Counter.
    """
    total = Counter()
    for c in counters:
        total.update(c)
    return total

def write_stats(counts: Counter, out_path: Path) -> None:
    """
    Write letter statistics to output file.

    Output format:
        <letter><TAB><count>

    Sorting:
    - Primary: descending frequency
    - Secondary: alphabetical order of letters
    """
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    with out_path.open("w", encoding="utf-8") as w:
        for letter, cnt in items:
            w.write(f"{letter}\t{cnt}\n")

def main() -> None:
    # Validate command-line arguments
    if len(sys.argv) < 3:
        print("Usage: create_letter_stat.py <stm_dir> <output_stat_file>")
        sys.exit(1)

    # Root directory containing STM files
    stm_root = Path(sys.argv[1])

    # Output statistics file
    out_path = Path(sys.argv[2])

    # Collect all STM files recursively
    files = list(iter_stm_files(stm_root))

    # If no STM files found, write empty stats and exit
    if not files:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_stats(Counter(), out_path)
        return

    # Determine number of worker processes
    cpu = os.cpu_count() or 1
    file_count = len(files)
    workers = min(MAX_PROCESSES, cpu, file_count)

    counters = []

    # Process STM files in parallel
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_stm_file, p): p for p in files}
        for fut in as_completed(futures):
            _path_str, ctr = fut.result()
            counters.append(ctr)

    # Merge per-file counters into a global counter
    total = merge_counters(counters)

    # Ensure output directory exists and write results
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_stats(total, out_path)

if __name__ == "__main__":
    main()
