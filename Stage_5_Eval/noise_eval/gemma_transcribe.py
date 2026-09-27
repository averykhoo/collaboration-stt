"""Benchmark transcripts from a Gemma 4 audio model, on the same speech segments as the Stage 5 oracle runs.

Usage (collaboration-stt env; needs transformers, pillow and torchvision==0.23.0 from the
PyTorch CPU index, to keep torch at 2.8):
    python noise_eval/gemma_transcribe.py MANIFEST.csv ORACLE_SEGMENTS.json OUT_DIR
        [--model google/gemma-4-E2B-it] [--ids id1,id2,...] [--threads 4]

Only the E2B, E4B and 12B checkpoints take audio; the hosted 26B-A4B and 31B
models on the Gemini API reject it ("Audio input modality is not enabled",
2026-09-25). Audio clips are limited to 30 s, so each oracle (clean-audio VAD)
segment is decoded separately, exactly as whisper_transcribe.py does. The
prompt is the model card's ASR prompt, decoding is greedy, and the output goes
through gemini_transcribe.normalise.

--dtype defaults to float32. On the laptop CPU (i7-1365U, no native bf16), fp32 E2B
was 3.4-4x faster than bfloat16 (2026-09-25), but needs about 20 GB of RAM
against about 12 GB. The two dtypes do NOT give identical text: only 131 of
210 segments matched on the 5 files run both ways (2026-09-27). Their accuracy
is close (clean WER 43.2% fp32 vs 43.9% bf16), but never mix them in one
comparison. The dtype is recorded in raw/<id>.json, and a resumed file must
use the same one.

Outputs:
  OUT_DIR/raw/<id>.json  model, prompt, raw text and seconds per segment
  OUT_DIR/<id>.txt       one normalised line per segment, scored by score.py

Slow on a laptop CPU, so progress is saved after every segment and a rerun
resumes mid-file.
"""
import argparse
import csv
import json
import os
import sys
import time

import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemini_transcribe import normalise  # noqa: E402

SR = 16000
PROMPT = ("Transcribe the following speech segment in its original language. Follow these specific "
          "instructions for formatting the answer:\n"
          "* Only output the transcription, with no newlines.\n"
          "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, "
          "and write 3 instead of three.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("oracle_segments")
    p.add_argument("out_dir")
    p.add_argument("--model", default="google/gemma-4-E2B-it")
    p.add_argument("--ids")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    a = p.parse_args()

    from transformers import AutoModelForMultimodalLM, AutoProcessor

    torch.set_num_threads(a.threads)
    with open(a.oracle_segments) as f:
        segs = json.load(f)["segments"]
    with open(a.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if a.ids:
        want = a.ids.split(",")
        rows = sorted((r for r in rows if r["id"] in want), key=lambda r: want.index(r["id"]))
    os.makedirs(os.path.join(a.out_dir, "raw"), exist_ok=True)

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForMultimodalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).eval()
    print(f"{len(rows)} files x {len(segs)} segments, model {a.model}, loaded in {time.time() - t0:.0f}s",
          flush=True)

    for r in rows:
        final = os.path.join(a.out_dir, r["id"] + ".txt")
        if os.path.exists(final):
            continue
        raw_path = os.path.join(a.out_dir, "raw", r["id"] + ".json")
        raw = json.load(open(raw_path, encoding="utf-8")) if os.path.exists(raw_path) else {
            "model": a.model, "prompt": PROMPT, "wav": r["wav"], "segments": segs,
            "texts": [], "seconds": [], "threads": a.threads, "dtype": a.dtype}
        assert (raw["model"], raw["prompt"], raw["segments"], raw["dtype"]) == (a.model, PROMPT, segs, a.dtype), \
            f"{raw_path} was made with other settings"
        x, sr = sf.read(r["wav"], dtype="float32")
        assert sr == SR and x.ndim == 1, f"{r['wav']}: expected {SR} Hz mono, got {sr} Hz {x.shape}"
        for b, e in segs[len(raw["texts"]):]:
            t = time.time()
            msgs = [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "audio", "audio": x[int(b * SR):int(e * SR)]}]}]
            inputs = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                              add_generation_prompt=True)
            n = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
            raw["texts"].append(proc.decode(out[0][n:], skip_special_tokens=True))
            raw["seconds"].append(round(time.time() - t, 1))
            with open(raw_path + ".part", "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False)
            os.replace(raw_path + ".part", raw_path)
            print(f"{r['id']} segment {len(raw['texts'])}/{len(segs)} ({e - b:.1f}s audio) "
                  f"in {raw['seconds'][-1]:.0f}s", flush=True)
        with open(final + ".part", "w", encoding="utf-8") as f:
            for t in raw["texts"]:
                f.write(normalise(t) + "\n")
        os.replace(final + ".part", final)
        print(f"{r['id']} done in {sum(raw['seconds']):.0f}s", flush=True)


if __name__ == "__main__":
    main()
