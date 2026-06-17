#!/usr/bin/env python3
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: concat_txts.py <in_dir> <out.txt>")
        sys.exit(1)

    in_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    # Find all .txt files and sort numerically if possible
    txt_files = sorted(
        in_dir.glob("*.txt"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
    )

    if not txt_files:
        print(f"No .txt files found in {in_dir}")
        sys.exit(1)

    with open(out_path, "w", encoding="utf-8") as outfile:
        for txt_file in txt_files:            
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read().rstrip()
            outfile.write(text + "\n")

    print(f"✅ Wrote {len(txt_files)} files into {out_path}")

if __name__ == "__main__":
    main()
