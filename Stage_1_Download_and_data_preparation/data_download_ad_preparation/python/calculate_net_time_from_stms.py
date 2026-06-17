# Usage:
# python python/calculate_net_time_from_stm_mp.py <in_stm_dir> <out_net_file.txt>

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import codecs

# ---------------------------------------------------------------------
# Return list of files in a directory with a given extension
# ---------------------------------------------------------------------
def dir_get_file_list(input_dir, my_extension):
    input_list = os.listdir(input_dir)          # list all entries in directory
    out_list = [i for i in input_list if i.endswith(my_extension)]  # filter by extension
    return out_list

# ---------------------------------------------------------------------
# Open a UTF-8 STM file for reading (handles BOM if present)
# ---------------------------------------------------------------------
def my_reader(in_path):
    fid = codecs.open(in_path,'r','utf-8-sig')
    return fid

# ---------------------------------------------------------------------
# Open output file for writing in UTF-8
# ---------------------------------------------------------------------
def my_writer(out_path):
    fid = codecs.open(out_path,'w','utf-8')
    return fid

# ---------------------------------------------------------------------
# Write a single line to a file and append newline
# ---------------------------------------------------------------------
def write_line(fid, str):
    fid.write(str + "\n")

# ---------------------------------------------------------------------
# Maximum number of parallel worker processes
# ---------------------------------------------------------------------
MAX_PROCESSES = 200

# ---------------------------------------------------------------------
# Compute total spoken duration (in seconds) for a single STM file
# ---------------------------------------------------------------------
def _file_total_seconds(cur_stm_path: str) -> float:
    """Return total spoken time in seconds for a single STM file."""
    total = 0.0
    fid = my_reader(cur_stm_path)

    # Iterate over STM lines
    for line in fid:
        line = line.strip()

        # Preserve original behavior: stop on first empty line
        if line == "":
            break

        parts = line.split(' ')

        # STM format:
        # <filename> <channel> <speaker> <start> <end> <label> [transcript...]
        # We only use start and end timestamps
        start = float(parts[3])
        end = float(parts[4])

        # Accumulate segment duration
        total += (end - start)

    return total

# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def main():
    # Validate command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python calculate_net_time_from_stm_mp.py <in_stm_dir> <out_net_file.txt>")
        sys.exit(1)

    in_dir = sys.argv[1]         # directory containing STM files
    out_net_file = sys.argv[2]   # output file path

    # Get list of STM filenames (not full paths)
    stm_files = dir_get_file_list(in_dir, '.stm')

    # If no STM files exist, write zero and exit
    if not stm_files:
        w = my_writer(out_net_file)
        write_line(w, "TOTAL TIME IN HOURS IS : 0.0")
        w.close()
        return

    # Limit number of worker processes
    max_workers = min(MAX_PROCESSES, len(stm_files))

    total_seconds = 0.0

    # Convert filenames to absolute paths once
    abs_paths = [os.path.join(in_dir, f) for f in stm_files]

    # Process STM files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_file_total_seconds, p): p for p in abs_paths}

        # Collect results as workers finish
        for fut in as_completed(futures):
            total_seconds += fut.result()

    # Convert seconds to hours
    total_hours = total_seconds / 3600.0

    # Write final result
    w = my_writer(out_net_file)
    write_line(w, "TOTAL TIME IN HOURS IS : " + str(total_hours))
    w.close()

# ---------------------------------------------------------------------
# Script entry guard
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
