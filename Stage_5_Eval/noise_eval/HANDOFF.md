# HANDOFF — Stage 5 noise/reverb evaluation and baselines

This is the volatile companion to the durable docs: what is true now, open items, and facts that
are not derivable from the code. Written 2026-09-28.
- `README.md`: index of experiments and code.
- `METHODS.md`: how it works and why.
- `RESULTS.md`: the numbers.
- `../RUNBOOK.md`: commands.

## State (2026-09-28)

- **Full grid (71 files) done:**
  - our model: oracle and noisy VAD;
  - Gemini 3.5 Transcribe;
  - Gemini 3.5 Transcribe Live;
  - Gemma 4 E2B (fp32);
  - Mesolitica Whisper turbo with fallback.

  Evidence is in `results_grid/`. The consolidated table is in `RESULTS.md` → "Full-grid
  comparison".
- **Draw-0 only (5 files):** Gemini 3 / 3.5 / 3.6 / 3.8 Flash, Flash-Lite, Whisper greedy, and
  Gemma E2B bf16. Evidence is in `results_2026-09-25/`.
- **Abandoned by the user:**
  - Gemini 3.6 Flash at 6/71: 503 rejections used up the daily quota.
  - Gemini 3.8 Flash at 22/71: 12 of the 22 outputs were blocked.

## Open items, ranked

1. **Gemma 4 12B QAT (q4_0) via llama.cpp: step 1 done on 2026-09-29, and the result is poor.**
   See `results_2026-09-29_gemma12b_probe/NOTES.md`.
   - `llama-mtmd-cli` aborts; `llama-server` works.
   - A 15 s segment takes 90–120 s.
   - On the clean clip it transcribes Malay as phonetic English-ish gibberish, adds unrequested
     translations, and loops when given a Malay-primed prompt.
   - **Blocked on a user decision:** either download 12B bf16 (about 24 GB; about 24 GB RAM) to
     tell the model's own quality apart from the port and quantisation, or drop the 12B and fall
     back to E4B.

   The original plan is kept below for reference.
   - **Ready:**
     - `.scratch/tools/llama.cpp/` holds llama.cpp build `b11205` (win-cpu-x64), including
       `llama-mtmd-cli.exe`;
     - `google/gemma-4-12B-it-qat-q4_0-gguf` (the model plus `mmproj-…gguf`) is in the Hugging
       Face cache.
   - **Risk:** llama.cpp's `docs/multimodal.md` lists Gemma 4 audio only for E2B/E4B. The 12B,
     whose audio path has no separate encoder, is not listed, so audio may not load.
   - **Step 1:** time one oracle segment (the audio flag is in `llama-mtmd-cli --help`) and check
     that the output is speech, not a description.
   - **If it works:** run clean plus a couple of noise-only and reverb files. Check fidelity
     against 12B bf16 via transformers on a few segments: the int8 Whisper experience says
     quantised builds must be checked.
   - **Fallback:** E4B via llama.cpp (documented as audio-capable), or E4B bf16 via
     `gemma_transcribe.py` (about 16 GB RAM, slow).
   - **Cost reference:** full-precision 12B does not fit in fp32 (about 48 GB). In bf16 it is
     estimated at 8–10 h per file on this laptop.
2. **Listen to the reference.** Line 2 of `gt_mesolitica.txt` ("tom melihat mary memecahkan kaca
   jendela") is produced by no system, and 87 reference tokens appear in no system's clean
   output. Listen to the first ~30 s of the clean clip and fix the reference if the text is
   absent. Every absolute WER moves if it changes.
3. **A second, neutral clip.** Suggested: a ~10-minute FLEURS test subset, plus the 3 m
   re-recording from Stage 3 (`rerecorded.wav`, real far-field, 30 min; our model scores 37.5%).
   - `score.py` needs multi-clip support first: about half a day.
   - The full grid costs about 70 CPU-hours per clip for Gemma plus Whisper.
4. **No-augmentation baseline for our model.** Retrain the Stage 4 recipe with
   `--enable-musan false` (about 7 GPU-days). The alternative is Mesolitica's
   `conformer-medium-mixed` vs `-mixed-augmented` pair; it needs `malaya-speech` and was not tried.
5. **Rotate both Gemini API keys.** Both were pasted into chat. They live in
   `~/.gemini_api_key` and `~/.gemini_api_key_2`, never in the repo.

## Facts not derivable from the code

- **Hardware:** i7-1365U laptop, 34 GB RAM, shared with other sessions. Run one CPU model at a
  time. Gemma fp32 needs about 20 GB, Whisper about 4.5 GB.
- **Gemma dtype:** fp32 is 3.4–4× faster than bf16 on this CPU (no native bf16), but the two
  dtypes differ on 38% of segments. Never mix them in one comparison.
- **Gemini free-tier quirks (2026-09):**
  - quotas are per key and per model;
  - HTTP 503 rejections count against the daily quota;
  - `gemini-2.5-flash` returns 404 for new keys;
  - the hosted Gemma 4 26B and 31B models reject audio.
- **The WER tool is not strict edit distance.** `wer.py` is the project's official scorer. It
  counts +1 to +11 errors per file more than Levenshtein, and was deliberately kept.
- **Our model was trained with MUSAN noise** (icefall's `--enable-musan` default), with no
  reverb. That explains its profile: acceptable on noise, collapsing on reverb.

## Scratch left behind

`.scratch/grid/`, `.scratch/whisper/`, `.scratch/gemini/`, `.scratch/gemma/` and `.scratch/review/`
hold working copies and logs. Everything that matters is tracked under `results_*`. The
`.scratch/tools/` downloads are kept for open item 1.
