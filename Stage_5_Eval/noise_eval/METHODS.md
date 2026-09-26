# Noise-eval methods: code map, data flow and process decisions

This file explains how the noise evaluation and its baselines fit together, and why each process
decision was made. It is written for someone checking the work for correctness.
- **Results:** `RESULTS.md`.
- **Commands:** `../RUNBOOK.md`.
- **Per-script detail:** each script's docstring.

## 1. Data flow

```
mesolitica.wav + gt_mesolitica.txt                 (exported/Stage5.tar.gz; 592 s, 101 utterances, 1,046 words)
  │
  ├─ augment.py ──► wav/<id>.wav + manifest.csv     (15 cells: RT60 bucket × SNR, × 5 seeded draws = 70 files)
  │
  ├─ VAD on the clean clip ──► oracle.segments.json  (42 segments, 532.5 s, each under 28.2 s)
  │
  ├─ recognisers, one output dir each: <id>.txt (+ raw/)
  │     transcribe_many.py         our Stage 5 model, noisy VAD or --oracle-segments
  │     whisper_transcribe.py      Mesolitica Whisper turbo, per oracle segment
  │     gemma_transcribe.py        Gemma 4 E2B, per oracle segment
  │     gemini_transcribe.py       Gemini generateContent, one request per file (oracle span)
  │     gemini_live_transcribe.py  Gemini Live WebSocket, streamed oracle span, server VAD
  │
  └─ scoring
        score.py        corpus WER per cell + paired two-level bootstrap CIs (full sets only)
        score_files.py  per-file WER, no CIs (partial or retry sets)
        vad_stats.py    how noisy-VAD segments differ from the oracle segments
```

## 2. Code map

| File | Role | Key invariant or check |
|---|---|---|
| `augment.py` | Drives FAST `FASTSynthesisTemplate`: room IR + stationary noise at a target SNR; everything else off | Seeded with `zlib.crc32`. Draw *d* reuses the same noise clip and slice seed across cells, and the same IR across SNRs within a bucket. Checks in `RESULTS.md` → Evidence. |
| `transcribe_many.py` | Loads our model once and decodes many files with `predict.py`'s `load_wav` / `segment` / `decode` | The clean output is byte-identical to the original GPU run. |
| `../predict.py` | Our model: VAD → Zipformer transducer, greedy | Skips only the segments that would crash (<400 samples or <9 fbank frames). |
| `../wer.py` | The original WER tool (Stage 2) | Column-0 traceback fix; the clean stat file is byte-identical to the original. |
| `wer_utt.py` | Reuses `wer.py`'s alignment but charges each error to a reference utterance | Asserts that per-utterance sums equal `wer.py` totals; sabotage-tested. |
| `score.py` | Cell WER plus bootstrap; `--vs` gives paired differences | Needs `<id>.txt` for every manifest row plus `clean.txt`. |
| `score_files.py` | Per-file WER; outputs over 5,000 words become "runaway" and are not aligned | Its scores match `score.py`'s per-run CSV. |
| `gemini_transcribe.py` | Gemini REST. Owns `normalise()`, shared by every non-native runner | Resumable. Raw responses are kept. Exit code 3 = daily quota spent. |
| `gemini_live_transcribe.py` | Gemini Live: paced PCM stream; final `inputTranscription` per turn | Every server message is logged. Failed attempts are kept as `.fail<N>.jsonl`. |
| `whisper_transcribe.py` | HF Whisper fp32 per segment; `--fallback` = standard temperature fallback | Revision and settings go in `raw/`. The int8 `ct2` backend is kept but not faithful. |
| `gemma_transcribe.py` | Gemma 4 per segment, model-card ASR prompt, greedy | Saves after every segment. The dtype is recorded and checked on resume. |

## 3. Scoring: what one WER number means

- **Tokenisation.** Every hypothesis is read with `read_hyp_words`: all lines are joined into one
  word sequence, `<unk>` is removed, and the text is split on whitespace. Line structure (one
  line per segment, chunk or turn) therefore has no effect on the score. That matters because
  the systems segment differently.
- **Alignment.** The whole hypothesis is aligned against the whole reference as one sequence, as
  `wer.py` does, with the three `allowed_*` lists applied. WER = (S+I+D)/(M+S+D).
- **Normalisation.** Our model emits reference-style text natively. Every other system's text
  goes through `gemini_transcribe.normalise`:
  - lowercase;
  - thousand separators removed;
  - digit runs over 12 characters dropped as loop output;
  - numbers spelled out in Malay;
  - `-_/` turned into spaces;
  - punctuation stripped, keeping word-internal apostrophes.

  Arabic script is **not** transliterated. It counts as errors against the romanised reference,
  which costs Whisper 26 words on clean.
- **Point estimate.** Per cell, counts are pooled over all draws and utterances.
- **Confidence intervals.** 10,000 bootstrap replicates, seed 0, two levels:
  1. Reference utterances are resampled with multinomial weights. Each replicate uses the same
     weights for every cell and every system, which is what makes `--vs` differences paired.
  2. Within each cell, the five draws are also resampled.

  The clean row has one run, so its CI reflects utterances only. With one draw per cell (the
  draw-0 tables), the draw level does nothing.
- **Insertion attribution.** Each insertion is charged to the utterance of the last reference
  word before it (utterance 0 if none).

## 4. Process decisions and why

1. **Oracle segments for the recogniser comparison.** The clean clip's VAD segments are applied
   to every noisy file, so recognisers are compared on identical speech. For our model, noisy
   VAD was also run: it is within about 2 pp of oracle everywhere (`RESULTS.md`, VAD cost), so
   oracle numbers are a fair stand-in for end to end.
2. **Whisper and Gemma decode each segment separately.** Every segment is under 30 s, so each
   fits one Whisper window and Gemma's 30 s audio limit.
3. **Gemini gets the whole file.** The oracle span is 0.0–592.8 s, so the Gemini systems
   receive everything and segment it themselves. They are end to end by construction.
4. **Greedy decoding, temperature 0, is the default for every system.** That matches our
   model's greedy search.
5. **Whisper's standard fallback is the fair Whisper baseline.** Pure greedy loops on noisy
   audio (e.g. "eh" × 444 on a 1.2 s segment). Stock Whisper uses temperature fallback, and so
   did the Stage 3 baseline. Fallback sampling is seeded (`torch.manual_seed(0)`), so reruns
   reproduce.
6. **Retries never replace results.** A failed request that was re-sent with identical
   settings is reported as a second attempt next to the first. A setting change such as
   `--thinking low` becomes its own row. Transient failures (HTTP 503, files not run) were
   completed, because they are not model outputs. Empty, blocked and runaway outputs **are**
   model outputs and score as they are.
7. **Runaway outputs are flagged, not truncated.** Truncating would invent a score. Such files
   are marked ✗ and excluded from `score.py` runs, which need full sets.
8. **Gemma runs in fp32.** It was 3.4–4× faster than bf16 on this CPU, with identical text on
   the segments compared. The d0 files exist in both dtypes (`results_2026-09-25/gemma` for
   bf16, `results_grid` for fp32 once done), which checks the equivalence on 5 × 42 segments.
9. **The int8 Whisper result was rejected**, not averaged in. It matched fp32 on only 13 of 42
   clean segments.
10. **Grid ids run draw-major** (`ids_by_draw.txt`). A partial grid then covers every cell
    evenly.

## 5. Known limitations

- **One clip, 1,046 words.** Absolute CIs are wide. Rely on the paired differences.
- **The test clip is home ground.** It comes from Mesolitica, and so does the training data of
  both our model and Mesolitica's Whisper.
- **Mesolitica spelling.** The reference follows Mesolitica conventions, which our training data
  shares. Some errors for other systems are spelling style (e.g. "di kalangan").
- **Only 4 distinct IRs in the 0.8s bucket.** Draws 0 and 2 share one, so that bucket's CI is
  slightly optimistic.
- **Our model was trained with MUSAN noise** (icefall's `--enable-musan` default) and no reverb.
  A no-augmentation baseline has not been built.
- **Gemini models change without notice.** Their results hold for the run dates given.
- **Timings are wall clock on a shared laptop.** Some runs overlapped other jobs, as noted where
  it happened.
