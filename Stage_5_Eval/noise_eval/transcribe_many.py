"""Transcribe many WAVs with the Stage 5 model, loading it once.

Usage (collaboration-stt env, any cwd):
    python noise_eval/transcribe_many.py MANIFEST.csv OUT_DIR [--oracle-segments CLEAN.wav]

MANIFEST.csv needs columns `id` and `wav` (absolute path). Each file's
transcript goes to OUT_DIR/<id>.txt, written exactly as predict.py writes it
for a one-line path list, and the segments decoded go to
OUT_DIR/<id>.segments.json. Existing transcripts are skipped, so the run is
resumable.

--oracle-segments: run predict.py's VAD segmentation once on CLEAN.wav and
decode every file with those segments instead of its own VAD. The augmented
files are time-aligned with the clean clip, so this isolates the recogniser
from VAD errors under noise.
"""
import argparse
import csv
import json
import os
import sys
import time

STAGE5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("out_dir")
    p.add_argument("--oracle-segments", metavar="CLEAN_WAV")
    a = p.parse_args()

    manifest, out_dir = os.path.abspath(a.manifest), os.path.abspath(a.out_dir)
    oracle = os.path.abspath(a.oracle_segments) if a.oracle_segments else None
    os.makedirs(out_dir, exist_ok=True)
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    todo = [r for r in rows if not os.path.exists(os.path.join(out_dir, r["id"] + ".txt"))]
    print(f"{len(rows)} in manifest, {len(todo)} to transcribe", flush=True)
    if not todo:
        return

    # predict.py resolves the VAD checkpoint and its imports relative to Stage_5_Eval,
    # and parses sys.argv in AudioTool.__init__.
    os.chdir(STAGE5)
    sys.path.insert(0, STAGE5)
    sys.argv = ["predict.py", "--wav_pathes", "unused", "--out_predict", "unused"]
    import torch
    from predict import AudioTool
    from k2_scripts.gen_utils import my_writer, write_line

    tool = AudioTool()
    tool.load_model("cuda" if torch.cuda.is_available() else "cpu")

    fixed = None
    if oracle:
        fixed = tool.segment(*tool.load_wav(oracle))
        with open(os.path.join(out_dir, "oracle.segments.json"), "w") as f:
            json.dump({"source": oracle.replace("\\", "/"), "segments": fixed}, f)
        print(f"oracle: {len(fixed)} segments from {oracle}", flush=True)

    for n, r in enumerate(todo, 1):
        t0 = time.time()
        wav, sr = tool.load_wav(r["wav"])
        segs = fixed if fixed is not None else tool.segment(wav, sr)
        hyps = tool.decode(wav, sr, segs)
        final = os.path.join(out_dir, r["id"] + ".txt")
        with open(os.path.join(out_dir, r["id"] + ".segments.json"), "w") as f:
            json.dump(segs, f)
        w = my_writer(final + ".part")
        for h in hyps:
            write_line(w, h)
        w.close()
        os.replace(final + ".part", final)  # only complete transcripts count as done
        print(f"[{n}/{len(todo)}] {r['id']} {len(segs)} segs {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
