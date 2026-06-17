#!/usr/bin/env python3
"""
Extract text from Whisper output file.
Parses lines in format: [start end] text
Outputs raw text, one utterance per line.
"""
import sys
import re


def parse_whisper_file(path: str):
    """Parse whisper output and extract raw text."""
    sentences = []
    ts_re = re.compile(r"\[(\d+\.\d+)\s+(\d+\.\d+)\]\s+(.*)")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = ts_re.match(line)
            if not m:
                continue

            sentence_raw = m.group(3).strip()
            if sentence_raw:
                sentences.append(sentence_raw)

    return sentences


def main():
    if len(sys.argv) != 3:
        print("usage: python extract_whisper_text.py in_whisper.txt out_text.txt")
        sys.exit(1)

    in_txt = sys.argv[1]
    out_txt = sys.argv[2]

    sentences = parse_whisper_file(in_txt)

    with open(out_txt, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")

    print(f"[DONE] sentences: {len(sentences)}")
    print(f"[DONE] wrote: {out_txt}")


if __name__ == "__main__":
    main()
