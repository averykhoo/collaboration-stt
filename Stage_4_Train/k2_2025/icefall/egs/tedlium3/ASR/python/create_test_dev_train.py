#!/usr/bin/env python3
"""
Split STM files into test, dev, and train lists.

Usage:
    python create_test_dev_train.py <stm_dir> <output_dir>

- test.list  : first 10 files (without .stm extension)
- dev.list   : next 10 files (without .stm extension)
- train.list : all remaining files
"""

import os
import sys


def main():
    stm_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Collect and sort .stm files
    stm_files = sorted(
        f for f in os.listdir(stm_dir) if f.endswith(".stm")
    )

    # Strip extension
    names = [os.path.splitext(f)[0] for f in stm_files]

    test_names = names[:10]
    dev_names = names[10:20]
    train_names = names[20:]

    os.makedirs(output_dir, exist_ok=True)

    for filename, entries in [
        ("test.list", test_names),
        ("dev.list", dev_names),
        ("train.list", train_names),
    ]:
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            for name in entries:
                f.write(name + "\n")
        print(f"Wrote {len(entries)} entries to {path}")


if __name__ == "__main__":
    main()
