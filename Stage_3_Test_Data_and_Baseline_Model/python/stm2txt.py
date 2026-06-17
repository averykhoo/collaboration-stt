#!/usr/bin/env python3
"""
Convert STM file to flat text file.
STM format: file_id channel speaker_id start end label text...
Outputs one utterance per line.
"""
import sys


def stm_to_txt(stm_path: str, out_path: str):
    """Extract text from STM file and write to flat text file."""
    count = 0
    with open(stm_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line or line.startswith(";;"):
                continue
            
            # STM format: file_id channel speaker_id start end label text...
            parts = line.split(None, 6)  # Split into max 7 parts
            if len(parts) < 7:
                continue
            
            text = parts[6].strip()
            if text:
                fout.write(text + "\n")
                count += 1
    
    return count


def main():
    if len(sys.argv) != 3:
        print("usage: python stm2txt.py input.stm output.txt")
        sys.exit(1)

    stm_path = sys.argv[1]
    out_path = sys.argv[2]

    count = stm_to_txt(stm_path, out_path)

    print(f"[DONE] utterances: {count}")
    print(f"[DONE] wrote: {out_path}")


if __name__ == "__main__":
    main()
