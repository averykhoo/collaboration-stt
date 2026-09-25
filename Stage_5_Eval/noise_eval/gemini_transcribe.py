"""Benchmark transcripts from a Gemini transcription model, on the same audio as the Stage 5 runs.

Usage (collaboration-stt env):
    python noise_eval/gemini_transcribe.py MANIFEST.csv ORACLE_SEGMENTS.json OUT_DIR
        [--model gemini-3.5-transcribe] [--chunk-s 0] [--ids id1,id2,...] [--prompt TEXT]

For general models such as gemini-3.5-flash-lite, pass --prompt (see PROMPT_MALAY)
and a shorter --min-interval. Those runs also use temperature 0.

The API key is read from ~/.gemini_api_key (or --key-file). Never put it in the repo.

--thinking LEVEL sets generationConfig.thinkingConfig.thinkingLevel for models
that think. On 2026-09-25, with the default level, gemini-3.5-flash and
gemini-3-flash-preview each spent about 63K thinking tokens on one file and
returned nothing usable.

Each file is cut into chunks at the oracle (clean-audio VAD) segment boundaries,
grouping consecutive segments while the chunk spans at most --chunk-s seconds.
Both systems therefore hear the same speech spans. The default, --chunk-s 0,
sends one chunk per file (first to last oracle segment), because the free tier
of gemini-3.5-transcribe allows only 25 requests per day (as of 2026-09-24).
Chunks go to generateContent as FLAC: inline when small, otherwise through the
Files API. Uploaded files are deleted straight after use.

Outputs:
  OUT_DIR/raw/<id>.json  one raw response per chunk, the untouched record
  OUT_DIR/<id>.txt       one normalised line per chunk, scored by score.py

Runs are resumable: finished files are skipped, and finished chunks are reused
from raw/. Requests are spaced --min-interval seconds apart. A 429 on a
per-minute quota waits and retries. A 429 on a per-day quota saves progress and
exits with code 3.
"""
import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

import soundfile as sf

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
UPLOAD = "https://generativelanguage.googleapis.com/upload/v1beta/files"
FILES = "https://generativelanguage.googleapis.com/v1beta"

_ONES = ["", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "lapan", "sembilan"]


def malay_number(n: int) -> str:
    """Integer to Malay words, e.g. 1995 -> 'seribu sembilan ratus sembilan puluh lima'."""
    if n == 0:
        return "kosong"
    parts = []
    for size, unit, single in ((10 ** 9, "bilion", "satu bilion"), (10 ** 6, "juta", "sejuta"),
                               (1000, "ribu", "seribu")):
        q, n = divmod(n, size)
        if q:
            parts.append(single if q == 1 else f"{malay_number(q)} {unit}")
    q, n = divmod(n, 100)
    if q:
        parts.append("seratus" if q == 1 else f"{_ONES[q]} ratus")
    if n >= 20:
        q, r = divmod(n, 10)
        parts.append(f"{_ONES[q]} puluh" + (f" {_ONES[r]}" if r else ""))
    elif n >= 11:
        parts.append("sebelas" if n == 11 else f"{_ONES[n - 10]} belas")
    elif n == 10:
        parts.append("sepuluh")
    elif n:
        parts.append(_ONES[n])
    return " ".join(parts)


def normalise(text: str) -> str:
    """Match the reference style: lowercase words, no punctuation, numbers spelled out."""
    text = text.lower()
    text = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", text)          # 1,000 / 1.000 -> 1000
    # A digit run longer than 12 is a repetition loop (seen: 64,667 digits of "0505..." up to
    # MAX_TOKENS), not a spoken number. Drop it; gemini_transcribe flags such responses.
    text = re.sub(r"\d{13,}", " ", text)
    text = re.sub(r"\d+", lambda m: f" {malay_number(int(m.group()))} ", text)
    text = re.sub(r"[-_/]", " ", text)                          # tiba-tiba -> tiba tiba
    text = re.sub(r"[^\w\s']|'(?!\w)|(?<!\w)'", " ", text)     # punctuation; keep word-internal '
    return " ".join(text.split())


def chunks_from(segments, max_s):
    out, cur = [], None
    for b, e in segments:
        if cur and e - cur[0] <= max_s:
            cur[1] = e
        else:
            if cur:
                out.append(cur)
            cur = [b, e]
    out.append(cur)
    return out


PROMPT_MALAY = ("Transcribe this audio verbatim. The speech is Malay (Malaysian), sometimes mixed with "
                "English. Write exactly the words spoken, in the language spoken. Do not translate, "
                "summarise, correct grammar or add speaker labels, timestamps or commentary. "
                "Output only the transcript.")

INLINE_MAX = 8_000_000  # bytes of FLAC; the whole inline request must stay under ~20 MB


def upload(key, flac: bytes) -> dict:
    """Resumable upload to the Gemini Files API. Returns the file resource."""
    start = urllib.request.Request(
        UPLOAD, data=json.dumps({"file": {"display_name": "noise_eval"}}).encode(),
        headers={"x-goog-api-key": key, "Content-Type": "application/json",
                 "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                 "X-Goog-Upload-Header-Content-Length": str(len(flac)),
                 "X-Goog-Upload-Header-Content-Type": "audio/flac"})
    url = urllib.request.urlopen(start, timeout=60).headers["X-Goog-Upload-URL"]
    put = urllib.request.Request(url, data=flac, headers={
        "X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"})
    f = json.load(urllib.request.urlopen(put, timeout=600))["file"]
    while f.get("state") == "PROCESSING":
        time.sleep(2)
        f = json.load(urllib.request.urlopen(urllib.request.Request(
            f"{FILES}/{f['name']}", headers={"x-goog-api-key": key}), timeout=60))
    if f.get("state") not in (None, "ACTIVE"):
        raise RuntimeError(f"upload ended in state {f.get('state')}")
    return f


def delete(key, f: dict) -> None:
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{FILES}/{f['name']}", method="DELETE", headers={"x-goog-api-key": key}), timeout=60)
    except urllib.error.URLError as err:  # files expire after 48 h anyway
        print(f"warning: could not delete {f['name']}: {err}", flush=True)


def call(model, key, flac: bytes, prompt=None, thinking=None):
    uploaded = None
    if len(flac) <= INLINE_MAX:
        part = {"inline_data": {"mime_type": "audio/flac", "data": base64.b64encode(flac).decode()}}
    else:
        uploaded = upload(key, flac)
        part = {"file_data": {"mime_type": "audio/flac", "file_uri": uploaded["uri"]}}
    try:
        body = {"contents": [{"parts": [part]}]}
        if prompt:
            body["contents"][0]["parts"].append({"text": prompt})
            body["generationConfig"] = {"temperature": 0.0}
        if thinking:
            body.setdefault("generationConfig", {})["thinkingConfig"] = {"thinkingLevel": thinking}
        return _generate(model, key, body)
    finally:
        if uploaded:
            delete(key, uploaded)


def _generate(model, key, body):
    req = urllib.request.Request(API.format(model=model), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    for attempt in range(4):
        try:
            # Thinking models (e.g. gemini-3.5-flash) took over 300 s on a 10-minute file.
            return json.load(urllib.request.urlopen(req, timeout=900))
        except urllib.error.HTTPError as err:
            if err.code == 429:
                body = err.read().decode()
                if "PerDay" in body or attempt == 3:
                    raise QuotaExhausted(body)
                m = re.search(r'"retryDelay":\s*"(\d+)', body)
                wait = int(m.group(1)) + 5 if m else 65
                print(f"per-minute quota hit, waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            if err.code >= 500 and attempt < 3:
                # 503 "high demand" spells lasted minutes on 2026-09-25; seconds-long backoff never cleared them.
                wait = 60 * 2 ** attempt
                print(f"HTTP {err.code}, retrying in {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {err.code}: {err.read().decode()[:500]}")
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


class QuotaExhausted(Exception):
    pass


def text_of(resp) -> str:
    parts = resp.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return " ".join(p.get("audioTranscription", {}).get("text", "") or p.get("text", "")
                    for p in parts if not p.get("thought"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("oracle_segments")
    p.add_argument("out_dir")
    p.add_argument("--model", default="gemini-3.5-transcribe")
    p.add_argument("--chunk-s", type=float, default=0.0, help="0 = one chunk per file")
    p.add_argument("--ids")
    p.add_argument("--prompt", help="text prompt for general models; 'malay' = PROMPT_MALAY")
    p.add_argument("--min-interval", type=float, default=65.0,
                   help="seconds between requests (3.5 Transcribe free tier: 10K tokens/min)")
    p.add_argument("--thinking", help="thinkingLevel, e.g. low; default: the model's own")
    p.add_argument("--key-file", default="~/.gemini_api_key")
    a = p.parse_args()

    key = open(os.path.expanduser(a.key_file)).read().strip()
    if a.prompt == "malay":
        a.prompt = PROMPT_MALAY
    with open(a.oracle_segments) as f:
        segs = json.load(f)["segments"]
    chunks = chunks_from(segs, a.chunk_s) if a.chunk_s > 0 else [[segs[0][0], segs[-1][1]]]
    with open(a.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if a.ids:
        want = set(a.ids.split(","))
        rows = [r for r in rows if r["id"] in want]
    os.makedirs(os.path.join(a.out_dir, "raw"), exist_ok=True)
    print(f"{len(rows)} files x {len(chunks)} chunks, model {a.model}", flush=True)

    sent, last = 0, 0.0
    for r in rows:
        final = os.path.join(a.out_dir, r["id"] + ".txt")
        if os.path.exists(final):
            continue
        raw_path = os.path.join(a.out_dir, "raw", r["id"] + ".json")
        raw = json.load(open(raw_path, encoding="utf-8")) if os.path.exists(raw_path) else {
            "model": a.model, "prompt": a.prompt, "thinking": a.thinking, "wav": r["wav"], "chunks": chunks,
            "responses": []}
        assert (raw["chunks"], raw["model"], raw.get("prompt"), raw.get("thinking")) == (
            chunks, a.model, a.prompt, a.thinking),             f"{raw_path} was made with other settings"
        x, sr = sf.read(r["wav"])
        for b, e in chunks[len(raw["responses"]):]:
            time.sleep(max(0.0, last + a.min_interval - time.time()))
            buf = io.BytesIO()
            sf.write(buf, x[int(b * sr):int(e * sr)], sr, format="FLAC")
            try:
                resp = call(a.model, key, buf.getvalue(), a.prompt, a.thinking)
            except QuotaExhausted as err:
                print(f"daily quota exhausted after {sent} requests: {err}", flush=True)
                sys.exit(3)
            last = time.time()
            sent += 1
            raw["responses"].append(resp)
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False)
        with open(final + ".part", "w", encoding="utf-8") as f:
            for resp in raw["responses"]:
                f.write(normalise(text_of(resp)) + "\n")
        os.replace(final + ".part", final)
        reasons = [resp.get("candidates", [{}])[0].get("finishReason") for resp in raw["responses"]]
        flag = "" if all(x == "STOP" for x in reasons) else f"  FLAG finishReason={reasons}"
        print(f"{r['id']} done ({sent} requests so far){flag}", flush=True)


if __name__ == "__main__":
    main()
