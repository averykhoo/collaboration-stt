"""Benchmark transcripts from the Gemini Live transcription model, on the same audio as the Stage 5 runs.

Usage (collaboration-stt env, needs aiohttp):
    python noise_eval/gemini_live_transcribe.py MANIFEST.csv ORACLE_SEGMENTS.json OUT_DIR
        [--model gemini-3.5-transcribe-live] [--ids id1,id2,...] [--speed 4] [--min-interval 60]

The API key is read from ~/.gemini_api_key (or --key-file). Never put it in the repo.

Each file is streamed over the Live API WebSocket (BidiGenerateContent) as
16 kHz 16-bit PCM in 0.5 s chunks, from the first to the last oracle segment,
the same span gemini_transcribe.py sends. Audio is paced at --speed times
real time. The server's own VAD cuts it into turns (its audioOffset values are
in audio time). After audioStreamEnd, the run for a file ends once the server
has been silent for --idle seconds.

The transcript is the final `inputTranscription` text of each server turn,
normalised with gemini_transcribe.normalise, one line per turn (see
turns_from_log). When a file ends mid-speech, the server's VAD may never close
the last turn, so its final never arrives and that text is lost. In the
2026-09-26 grid this truncated 8 of 71 files; the lost text was usually the
reference's last line. Streams now end with 2 s of silence so the turn can
close. A log that still ends inside a turn is reported as TRUNCATED, and that
file should be re-run. --rebuild re-derives every OUT_DIR/<id>.txt from the raw
logs without calling the API, and lists truncated files.

Free tier as of 2026-09-25: unlimited requests per day, 20K tokens per minute.
A 10-minute file is about 14.8K tokens. Sending a whole file unpaced closed the
socket with 1011 "Resource has been exhausted" (2026-09-25), so audio is paced
(about 1.5K tokens per minute of audio, times --speed) and files are spaced
--min-interval seconds apart.

Outputs:
  OUT_DIR/raw/<id>.jsonl  every server message, the untouched record
  OUT_DIR/<id>.txt        one normalised line per turn, scored by score.py

Runs are resumable: finished files are skipped. A file whose socket closes
abnormally is retried up to --retries times and never gets a .txt otherwise.
Each failed attempt's messages are kept as raw/<id>.fail<N>.jsonl.
"""
import argparse
import asyncio
import base64
import csv
import glob
import json
import os
import re
import sys
import time

import aiohttp
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemini_transcribe import normalise  # noqa: E402

URL = ("wss://generativelanguage.googleapis.com/ws/"
       "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")
SR = 16000


TAIL_SILENCE_S = 2.0


class StreamFailed(Exception):
    pass


def turns_from_log(messages):
    """Final `inputTranscription` per turn, in order; returns (turns, truncated).

    truncated is True when the log ends inside a turn: interim text arrived after the last
    final, so the stream stopped before the server finalised it. That is a client-side loss
    and the file should be re-run. Mid-stream turns with interim text but no final are left
    out on purpose: in the 2026-09-26 grid they were the server discarding hallucinations
    (Portuguese, Russian, Italian, Korean fragments) or a duplicate of the previous final.
    """
    turns, pending = [], False
    for d in messages:
        sc = d.get("serverContent", {})
        if "interimInputTranscription" in sc:
            pending = True
        if "inputTranscription" in sc:
            turns.append(sc["inputTranscription"].get("text", ""))
            pending = False
        if d.get("voiceActivity", {}).get("type") == "ACTIVITY_START":
            pending = False
    return turns, pending


def write_transcript(final, turns):
    with open(final + ".part", "w", encoding="utf-8") as f:
        for t in turns:
            f.write(normalise(t) + "\n")
    os.replace(final + ".part", final)


async def stream(key, model, pcm, raw_path, idle_s, speed):
    """Stream int16 PCM plus TAIL_SILENCE_S of zeros; return turns_from_log. Every message goes to raw_path."""
    messages = []
    with open(raw_path, "w", encoding="utf-8") as log:
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(URL, headers={"x-goog-api-key": key}, max_msg_size=0) as ws:
                await ws.send_json({"setup": {"model": f"models/{model}"}})
                m = await ws.receive(timeout=30)
                if m.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                    raise StreamFailed(f"setup: {m.type} {ws.close_code} {m.extra}")
                log.write(json.dumps(json.loads(m.data), ensure_ascii=False) + "\n")

                async def sender():
                    step, t0 = SR // 2, time.monotonic()
                    for i in range(0, len(pcm), step):
                        if speed > 0:
                            await asyncio.sleep(max(0.0, t0 + i / SR / speed - time.monotonic()))
                        await ws.send_json({"realtimeInput": {"audio": {
                            "data": base64.b64encode(pcm[i:i + step].tobytes()).decode(),
                            "mimeType": f"audio/pcm;rate={SR}"}}})
                    silence = bytes(2 * int(SR * TAIL_SILENCE_S))  # lets the server VAD close the last turn
                    await ws.send_json({"realtimeInput": {"audio": {
                        "data": base64.b64encode(silence).decode(), "mimeType": f"audio/pcm;rate={SR}"}}})
                    await ws.send_json({"realtimeInput": {"audioStreamEnd": True}})

                send = asyncio.create_task(sender())
                try:
                    while True:
                        try:
                            m = await ws.receive(timeout=idle_s)
                        except asyncio.TimeoutError:
                            if send.done():
                                break
                            continue
                        if m.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                            raise StreamFailed(f"socket {m.type}, close code {ws.close_code}: {m.extra}")
                        d = json.loads(m.data)
                        log.write(json.dumps(d, ensure_ascii=False) + "\n")
                        if "error" in d or "goAway" in d:
                            raise StreamFailed(json.dumps(d)[:500])
                        messages.append(d)
                finally:
                    send.cancel()
                send.result()  # the loop only ends once the sender is done; re-raise its error, if any
    return turns_from_log(messages)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("oracle_segments")
    p.add_argument("out_dir")
    p.add_argument("--model", default="gemini-3.5-transcribe-live")
    p.add_argument("--ids")
    p.add_argument("--min-interval", type=float, default=60.0,
                   help="seconds between file starts (free tier: 20K tokens/min)")
    p.add_argument("--idle", type=float, default=30.0,
                   help="after the audio is sent, stop once the server is silent this long")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--speed", type=float, default=4.0, help="times real time; 0 = unpaced")
    p.add_argument("--key-file", default="~/.gemini_api_key")
    p.add_argument("--rebuild", action="store_true",
                   help="rewrite every OUT_DIR/<id>.txt from raw/<id>.jsonl, no API calls")
    a = p.parse_args()

    if a.rebuild:
        for raw_path in sorted(glob.glob(os.path.join(a.out_dir, "raw", "*.jsonl"))):
            run_id = os.path.basename(raw_path)[:-len(".jsonl")]
            if re.search(r"\.(fail\d+|truncated)$", run_id):  # kept logs of superseded attempts
                continue
            with open(raw_path, encoding="utf-8") as f:
                turns, truncated = turns_from_log([json.loads(l) for l in f])
            write_transcript(os.path.join(a.out_dir, run_id + ".txt"), turns)
            print(f"{run_id}: {len(turns)} turns" + ("  TRUNCATED" if truncated else ""), flush=True)
        return

    key = open(os.path.expanduser(a.key_file)).read().strip()
    with open(a.oracle_segments) as f:
        segs = json.load(f)["segments"]
    span = (segs[0][0], segs[-1][1])
    with open(a.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if a.ids:
        want = a.ids.split(",")
        rows = sorted((r for r in rows if r["id"] in want), key=lambda r: want.index(r["id"]))
    os.makedirs(os.path.join(a.out_dir, "raw"), exist_ok=True)
    print(f"{len(rows)} files, span {span[0]:.1f}-{span[1]:.1f} s, model {a.model}", flush=True)

    last = 0.0
    for r in rows:
        final = os.path.join(a.out_dir, r["id"] + ".txt")
        if os.path.exists(final):
            continue
        x, sr = sf.read(r["wav"], dtype="int16")  # the eval files are 16-bit PCM already
        assert sr == SR and x.ndim == 1, f"{r['wav']}: expected {SR} Hz mono, got {sr} Hz {x.shape}"
        pcm = x[int(span[0] * SR):int(span[1] * SR)].astype("<i2")
        for attempt in range(a.retries + 1):
            time.sleep(max(0.0, last + a.min_interval - time.time()))
            last = time.time()
            raw_path = os.path.join(a.out_dir, "raw", r["id"] + ".jsonl")
            try:
                finals, truncated = asyncio.run(stream(key, a.model, pcm, raw_path, a.idle, a.speed))
                break
            except (StreamFailed, aiohttp.ClientError, asyncio.TimeoutError) as err:
                print(f"{r['id']} attempt {attempt + 1} failed after {time.time() - last:.0f}s: {err}", flush=True)
                os.replace(raw_path, raw_path.replace(".jsonl", f".fail{attempt + 1}.jsonl"))
        else:
            print(f"{r['id']} gave up", flush=True)
            continue
        write_transcript(final, finals)
        print(f"{r['id']} done in {time.time() - last:.0f}s, {len(finals)} turns{'  TRUNCATED' if truncated else ''}, "
              f"{sum(len(t.split()) for t in finals)} words", flush=True)


if __name__ == "__main__":
    main()
