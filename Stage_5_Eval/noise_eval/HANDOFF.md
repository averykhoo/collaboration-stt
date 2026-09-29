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
  - Gemma 4 12B QAT (q4_0) via llama.cpp, closed on 2026-09-29 after a 3-segment probe. See
    `results_2026-09-29_gemma12b_probe/NOTES.md`.
    - The output was poor: Malay came out as phonetic English-ish text, and a Malay-primed
      prompt made it loop.
    - It was slow: about 1 h per file.
    - The deciding bf16 check needs about 24 GB of RAM, which this 34 GB shared laptop cannot
      spare.
    - If a larger Gemma is wanted later, the untried fallback is E4B, via llama.cpp
      `llama-server` or via `gemma_transcribe.py` in bf16 (about 16 GB RAM).

## Open items, ranked

1. **Listen to the reference.** Line 2 of `gt_mesolitica.txt` ("tom melihat mary memecahkan kaca
   jendela") is produced by no system, and 87 reference tokens appear in no system's clean
   output. Listen to the first ~30 s of the clean clip and fix the reference if the text is
   absent. Every absolute WER moves if it changes.
2. **A second, neutral clip.** Suggested: a ~10-minute FLEURS test subset, plus the 3 m
   re-recording from Stage 3 (`rerecorded.wav`, real far-field, 30 min; our model scores 37.5%).
   - `score.py` needs multi-clip support first: about half a day.
   - The full grid costs about 70 CPU-hours per clip for Gemma plus Whisper.
3. **No-augmentation baseline for our model.** Retrain the Stage 4 recipe with
   `--enable-musan false` (about 7 GPU-days). The alternative is Mesolitica's
   `conformer-medium-mixed` vs `-mixed-augmented` pair; it needs `malaya-speech` and was not tried.
4. **Rotate both Gemini API keys.** Both were pasted into chat. They live in
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
hold working copies and logs. Everything that matters is tracked under `results_*`. `.scratch/tools/llama.cpp/`, `.scratch/gemma12b/` and the 12B GGUF in the HF cache (about 7 GB)
were only for the closed 12B probe and can be deleted.
