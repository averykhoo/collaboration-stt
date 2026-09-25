"""Benchmark transcripts from a Whisper model, on the same speech segments as the Stage 5 oracle runs.

Usage (collaboration-stt env, needs `pip install transformers`):
    python noise_eval/whisper_transcribe.py MANIFEST.csv ORACLE_SEGMENTS.json OUT_DIR
        [--model mesolitica/Malaysian-whisper-large-v3-turbo-v3] [--ids id1,id2,...]
        [--batch 4] [--threads 4] [--backend hf|ct2 --ct2-dir DIR] [--fallback]

Each oracle (clean-audio VAD) segment is decoded separately. They are all under
30 s, so each fits one Whisper window, and Whisper hears exactly the spans that
our model heard in oracle mode. Decoding is greedy, language forced to Malay
("ms"), task "transcribe", no timestamps. Output text goes through
gemini_transcribe.normalise, the same normalisation as the Gemini baselines.

--backend ct2 decodes with faster-whisper (CTranslate2, int8 on CPU) from a
converted model directory, made with:
    ct2-transformers-converter --model MODEL --output_dir DIR --quantization int8
        --copy_files tokenizer.json preprocessor_config.json
It uses the same greedy settings, with temperature fallback, the no-speech
filter and faster-whisper's own VAD all off, so each segment is decoded once.

--fallback turns on Whisper's standard temperature fallback: a segment whose
output is too repetitive or too unlikely (mean log-prob < -1) is re-decoded at
temperature 0.2, 0.4, ... 1.0. The repetition test is gzip compression ratio
> 2.4 on the text for ct2, and the equivalent 1.35 on token ids for hf (the
value the transformers docs give). Without it, greedy decoding on noisy audio
falls into repetition loops ("eh eh eh ..." 444 times on a 1 s segment) that
dominate WER. The no-speech filter stays off, so no segment is dropped.
Sampling at T > 0 uses a fixed seed per run.

int8 is not a faithful stand-in for the fp32 model: on 2026-09-25 only 13 of 42
clean segments matched the hf fp32 output, with more repetition loops, for about
5% less time on the clean clip.

Outputs:
  OUT_DIR/raw/<id>.json  model, revision, raw text per segment and timing
  OUT_DIR/<id>.txt       one normalised line per segment, scored by score.py

Runs are resumable: finished files are skipped.
"""
import argparse
import csv
import json
import os
import sys
import time

import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemini_transcribe import normalise  # noqa: E402

SR = 16000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("oracle_segments")
    p.add_argument("out_dir")
    p.add_argument("--model", default="mesolitica/Malaysian-whisper-large-v3-turbo-v3")
    p.add_argument("--ids")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--backend", choices=["hf", "ct2"], default="hf")
    p.add_argument("--ct2-dir", help="converted model directory, for --backend ct2")
    p.add_argument("--fallback", action="store_true", help="Whisper's standard temperature fallback")
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(0)  # sampling only happens on fallback (T > 0)
    temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    with open(a.oracle_segments) as f:
        segs = json.load(f)["segments"]
    with open(a.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if a.ids:
        want = a.ids.split(",")
        rows = sorted((r for r in rows if r["id"] in want), key=lambda r: want.index(r["id"]))
    os.makedirs(os.path.join(a.out_dir, "raw"), exist_ok=True)

    t0 = time.time()
    if a.backend == "ct2":
        import ctranslate2
        from faster_whisper import WhisperModel
        ctranslate2.set_random_seed(0)  # sampling only happens on fallback (T > 0)
        model = WhisperModel(a.ct2_dir, device="cpu", compute_type="int8", cpu_threads=a.threads)
        revision = None
        fallback = dict(temperature=temperatures, compression_ratio_threshold=2.4,
                        log_prob_threshold=-1.0) if a.fallback else dict(temperature=0.0)

        def transcribe(pieces):
            texts = []
            for piece in pieces:
                out, _ = model.transcribe(piece, language="ms", task="transcribe", beam_size=1,
                                          **fallback, condition_on_previous_text=False,
                                          without_timestamps=True, vad_filter=False,
                                          no_speech_threshold=None)
                texts.append("".join(s.text for s in out).strip())
            return texts
    else:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        processor = WhisperProcessor.from_pretrained(a.model)
        model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float32).eval()
        revision = getattr(model.config, "_commit_hash", None)
        fallback = dict(temperature=tuple(temperatures), compression_ratio_threshold=1.35,
                        logprob_threshold=-1.0) if a.fallback else dict(do_sample=False)

        def transcribe(pieces):
            texts = []
            for i in range(0, len(pieces), a.batch):
                feats = processor.feature_extractor(pieces[i:i + a.batch], sampling_rate=SR,
                                                    return_tensors="pt").input_features
                with torch.inference_mode():
                    ids = model.generate(feats, language="ms", task="transcribe", num_beams=1,
                                         return_timestamps=False, **fallback)
                texts += processor.batch_decode(ids, skip_special_tokens=True)
            return texts
    print(f"{len(rows)} files x {len(segs)} segments, model {a.model}@{revision}, "
          f"backend {a.backend}, loaded in {time.time() - t0:.0f}s", flush=True)

    for r in rows:
        final = os.path.join(a.out_dir, r["id"] + ".txt")
        if os.path.exists(final):
            continue
        t0 = time.time()
        x, sr = sf.read(r["wav"], dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
        pieces = [x[int(b * SR):int(e * SR)] for b, e in segs]
        texts = transcribe(pieces)
        elapsed = time.time() - t0
        raw = {"model": a.model, "revision": revision, "wav": r["wav"], "segments": segs,
               "texts": texts, "seconds": round(elapsed, 1), "threads": a.threads,
               "backend": a.backend, "ct2_dir": a.ct2_dir, "fallback": a.fallback}
        with open(os.path.join(a.out_dir, "raw", r["id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)
        with open(final + ".part", "w", encoding="utf-8") as f:
            for t in texts:
                f.write(normalise(t) + "\n")
        os.replace(final + ".part", final)
        print(f"{r['id']} done in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
