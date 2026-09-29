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
- **This is not strict edit distance.** `wer.py::__global_alignment` maximises a score (match
  +2, insertion and deletion −0.75 each), not the minimum number of edits. It can report more
  errors than the minimum:
  - by 1 to 11 words (≤ 1.1 pp) on the transcripts checked on 2026-09-26: ours clean +1,
    Transcribe clean +3, Gemma clean +1;
  - most on insertion-heavy output, such as Whisper's looping 0.8s file (+11).

  It is the project's official Stage 2 scorer and is applied identically to every system, so
  it was kept; changing it would silently move every past number.
- **Normalisation.** Our model emits reference-style text natively. Every other system's text
  goes through `gemini_transcribe.normalise`:
  - lowercase;
  - thousand separators removed;
  - digit runs over 12 characters dropped as loop output;
  - numbers spelled out in Malay;
  - `-_/` turned into spaces;
  - punctuation stripped, keeping word-internal apostrophes.

  Some of the baselines' output can never match the reference, whatever they heard:
  - **Arabic script** is not transliterated. It counts as errors against the romanised
    reference, which costs Whisper 26 words on clean.
  - **Word-internal apostrophes** are kept, but the reference has none. "ka'ab", "ta'ala" and
    "qur'an" (the reference writes "quran") are errors: 12–35 tokens per baseline directory,
    ≤ 0.5 pp per file.
  - **Live sometimes joins words in CamelCase** ("kiriDalamOpa"), which `normalise` cannot
    split: 0–10 per file.

  These costs apply to the baselines only: our model's output never contains them.
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
4. **Greedy decoding is the default** wherever the API allows it, to match our model's greedy
   search:
   - Whisper and Gemma decode greedily.
   - The prompted Gemini Flash models run at temperature 0.
   - Gemini Transcribe (no prompt, no `generationConfig`) and Transcribe Live use the service's
     own decoding, which is not controllable.
5. **Whisper's standard fallback is the fair Whisper baseline.** Pure greedy loops on noisy
   audio (e.g. "eh" × 444 on a 1.2 s segment). Stock Whisper uses temperature fallback, and so
   did the Stage 3 baseline. Fallback sampling is seeded (`torch.manual_seed(0)`), but only once
   per process, so a rerun reproduces only if it processes the same files in the same order.
6. **Retries never replace results.** A failed request that was re-sent with identical
   settings is reported as a second attempt next to the first. A setting change such as
   `--thinking low` becomes its own row. Transient failures (HTTP 503, files not run) were
   completed, because they are not model outputs. Empty, blocked and runaway outputs **are**
   model outputs and score as they are.
7. **Runaway outputs are flagged, not truncated.** Truncating would invent a score. Such files
   are marked ✗ and excluded from `score.py` runs, which need full sets.
8. **Gemma runs in fp32 on the grid.** It was 3.4–4× faster than bf16 on this CPU. The two
   dtypes are **not** equivalent: on the 5 × 42 segments run both ways, only 131 of 210 match
   exactly. The first check covered only 3 segments, which all matched. Accuracy is close
   (clean 43.2% fp32 vs 43.9% bf16), but the draw-0 bf16 table and the fp32 grid are different
   runs and should not be mixed.
9. **The int8 Whisper result was rejected**, not averaged in. It matched fp32 on only 13 of 42
   clean segments.
10. **Grid ids run draw-major** (`ids_by_draw.txt`). A partial grid then covers every cell
    evenly.

## 5. Known limitations

- **One clip, 1,046 words.** Absolute CIs are wide. Rely on the paired differences.
- **The reference seems to contain text that is not in the audio.**
  - 87 of the 1,046 reference tokens (8.3%) are words that appear nowhere in any of six
    systems' clean outputs. That counts vocabulary: ours, Transcribe, Live, Whisper, Gemma and
    3.6 Flash.
  - By a stricter measure, 166 reference positions are matched by no system in alignment.
  - Line 2, "tom melihat mary memecahkan kaca jendela", is produced by none of them, including
    the Gemini models, which hear the whole file.
  - Unless someone listens and the reference is corrected, every absolute WER has a floor of
    roughly 8 pp. That includes our 35.5% clean headline.
  - Paired differences are largely unaffected, because every system pays the same floor.
- **The bootstrap treats the 101 utterances as independent.** Errors actually cluster by VAD
  segment: reverb deletes whole segments, and one looping segment can add hundreds of
  insertions to a single utterance. The CIs for reverb cells and for looping systems are
  probably too narrow. A block bootstrap over segments would be more honest; it has not been
  done.
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
