#!/usr/bin/env python3
import sys
import json

def iter_json_objects(path):
    """
    Yields JSON objects from a file containing
    pretty-printed JSON blocks separated by blank lines.
    """
    buf = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if buf:
                    yield json.loads("".join(buf))
                    buf = []
            else:
                buf.append(line)
        if buf:
            yield json.loads("".join(buf))


def main():
    if len(sys.argv) != 3:
        print("usage: python gt_json_to_txt.py inter/gt.json inter/gt.txt")
        sys.exit(1)

    in_json = sys.argv[1]
    out_txt = sys.argv[2]

    count = 0
    with open(out_txt, "w", encoding="utf-8") as out:
        for obj in iter_json_objects(in_json):
            text = obj.get("text_norm", "").strip()
            if not text:
                continue
            out.write(text + "\n")
            count += 1

    print(f"[DONE] wrote {count} lines to {out_txt}")


if __name__ == "__main__":
    main()
