# Stage 5 model under far-field noise and reverb

Run on 2026-09-24, on a laptop CPU (i7-1365U, 4 threads). All numbers in this file come from
that run. The tables regenerate from `results_2026-09-24/` using the commands at the end.

## What was tested

- **Speech.** The Stage 5 Malay test clip `mesolitica.wav` (592 s), scored against
  `gt_mesolitica.txt` (101 utterances, 1,046 reference words). Both come from
  `exported/Stage5.tar.gz`.
- **Augmentation.** Far-field Audio Synthesis Toolkit (FAST), `FASTSynthesisTemplate`, driven by
  `augment.py`. Only two stages are on:
  - **Room reverb:** a real IR from FAST's `room_IRs/<bucket>` folder, convolved with the speech.
    The direct-path onset is trimmed, so the output stays time-aligned.
  - **Stationary noise:** a DEMAND-style noise bed with seeded random slicing, mixed at the target
    SNR. The SNR is measured against the reverberant speech over the whole file.
  - **Off:** tempo/pitch (fixed at 1.0/0.0, so rubberband is bypassed), occlusion, device IR,
    Opus codec, phone lowpass and non-stationary noise. The noise bed itself is dry.
  - **Output level:** normalised to a 0.99 peak and saved as 16-bit PCM.
- **Grid.** RT60 bucket ∈ {none, 0.4s, 0.8s} × SNR ∈ {none, 20, 10, 5, 0} dB, giving 14 noisy
  cells plus clean. Each cell has 5 draws, 70 files in total.
- **Random draws.** Draw *d* uses the same noise clip and slice seed in every cell, and the same
  IR at every SNR within a bucket, so cells differ only in the setting.
  - Noise clips by draw: `OOFFICE-ch15`, `PRESTO-ch07` (restaurant), `PCAFETER-ch10`
    (cafeteria), `PSTATION-ch06` (station), `DWASHING-ch01`.
  - The bucket name is an upper bound. The measured T30 of the drawn IRs:

    | Bucket | Measured T30 of drawn IRs (s) |
    | --- | --- |
    | 0.4s | 0.32, 0.39, 0.36, 0.32, 0.21 |
    | 0.8s | 0.68, 0.64, 0.68, 0.61, 0.61 |

  - **Only 4 distinct IRs in the 0.8s bucket.** Draws 0 and 2 share the same IR
    (`room_021_v4_far_4.34m`), so that bucket's CI is slightly optimistic.
  - Full per-draw details are in `results_2026-09-24/manifest.csv`.
- **Two segmentation conditions.** Both decode with `predict.py`, greedy search.
  - **Noisy VAD:** the normal pipeline. The VAD runs on each noisy file.
  - **Oracle VAD:** the VAD segments computed on the *clean* clip (42 segments, 532.5 s), applied
    unchanged to every noisy file. This isolates the recogniser from VAD errors.

## Statistics

- **WER definition.** Corpus WER exactly as `wer.py` computes it: (S+I+D)/(M+S+D) over the whole
  clip, including the allowed-substitution lists.
- **Per-utterance attribution.** `wer_utt.py` charges each error to its reference utterance and
  asserts that the per-utterance sums reproduce `wer.py`'s totals.
- **95% CIs.** From a two-level paired bootstrap with 10,000 replicates:
  - Utterances are resampled, with the same resample used for every condition, so differences
    are paired.
  - Within each cell, the 5 draws are resampled as well.
  - The clean CI therefore reflects test-set sampling only, and the noisy CIs add augmentation
    randomness on top.
- **Width of the CIs.** Absolute WER CIs are wide because this is one ~1,000-word clip. The
  paired Δ columns are the precise comparisons.

## Results: oracle VAD (recogniser only)

| RT60 bucket | SNR (dB) | WER % [95% CI] | per-draw mean ± sd | ΔWER vs clean, pp [95% CI] |
|---|---|---|---|---|
| clean | clean | 35.5 [29.0, 42.3] | — | — |
| none | 20 | 36.3 [30.0, 42.9] | 36.3 ± 0.7 | +0.8 [-1.8, +3.3] |
| none | 10 | 41.8 [33.9, 50.1] | 41.8 ± 5.5 | +6.4 [+1.5, +12.4] |
| none | 5 | 47.6 [37.4, 58.5] | 47.6 ± 10.4 | +12.1 [+3.7, +22.0] |
| none | 0 | 56.0 [40.6, 72.2] | 56.0 ± 18.5 | +20.6 [+6.2, +36.7] |
| 0.4s | none | 76.0 [66.9, 84.2] | 76.0 ± 6.6 | +40.6 [+32.0, +49.1] |
| 0.4s | 20 | 77.1 [68.0, 85.3] | 77.1 ± 7.1 | +41.6 [+32.6, +50.4] |
| 0.4s | 10 | 81.4 [71.7, 89.4] | 81.4 ± 8.8 | +45.9 [+36.4, +55.1] |
| 0.4s | 5 | 84.7 [73.6, 94.0] | 84.7 ± 11.7 | +49.2 [+37.9, +60.0] |
| 0.4s | 0 | 87.1 [75.6, 96.8] | 87.1 ± 12.5 | +51.6 [+39.8, +62.7] |
| 0.8s | none | 93.6 [90.6, 96.2] | 93.6 ± 0.8 | +58.1 [+51.1, +64.8] |
| 0.8s | 20 | 93.8 [90.9, 96.3] | 93.8 ± 0.9 | +58.4 [+51.5, +65.0] |
| 0.8s | 10 | 95.0 [91.9, 97.6] | 95.0 ± 2.2 | +59.6 [+52.6, +66.3] |
| 0.8s | 5 | 95.6 [92.5, 97.9] | 95.6 ± 2.5 | +60.2 [+53.1, +66.8] |
| 0.8s | 0 | 96.6 [93.5, 99.0] | 96.6 ± 2.9 | +61.1 [+53.9, +67.8] |

## Results: noisy VAD (end to end) vs oracle

The last column is noisy-VAD WER minus oracle-VAD WER, paired, so it shows what VAD errors cost.

| RT60 bucket | SNR (dB) | noisy-VAD WER % [95% CI] | ΔWER vs clean, pp | VAD cost, pp [95% CI] |
|---|---|---|---|---|
| none | 20 | 35.8 [29.6, 42.4] | +0.3 | -0.5 [-2.7, +1.9] |
| none | 10 | 40.7 [33.1, 48.6] | +5.2 | -1.1 [-3.7, +1.1] |
| none | 5 | 46.9 [36.2, 58.5] | +11.4 | -0.7 [-3.7, +2.1] |
| none | 0 | 56.4 [40.4, 73.0] | +21.0 | +0.4 [-2.0, +2.8] |
| 0.4s | none | 78.3 [69.2, 86.3] | +42.8 | +2.2 [+0.2, +4.6] |
| 0.4s | 20 | 78.8 [69.5, 87.0] | +43.3 | +1.7 [-0.4, +4.0] |
| 0.4s | 10 | 83.1 [72.6, 91.6] | +47.6 | +1.7 [-0.8, +4.8] |
| 0.4s | 5 | 86.5 [75.0, 96.0] | +51.0 | +1.8 [-0.1, +4.1] |
| 0.4s | 0 | 87.4 [75.2, 97.9] | +52.0 | +0.3 [-1.6, +2.3] |
| 0.8s | none | 95.2 [92.5, 97.3] | +59.7 | +1.6 [-0.4, +4.0] |
| 0.8s | 20 | 94.8 [92.3, 96.9] | +59.4 | +1.0 [-1.0, +3.3] |
| 0.8s | 10 | 95.9 [93.6, 97.7] | +60.4 | +0.8 [-0.9, +2.9] |
| 0.8s | 5 | 96.0 [92.6, 98.5] | +60.5 | +0.4 [-1.5, +2.1] |
| 0.8s | 0 | 96.7 [93.4, 99.0] | +61.2 | +0.1 [-1.5, +1.7] |

How the noisy VAD's segments drift from the oracle segments (`result_vadstats.md`):
- **Missed speech:** almost none. It still covers 97–100% of the oracle speech time.
- **Extra time:** it pulls in noise-only stretches, a mean of 6–34 s depending on the cell and up
  to about 59 s for the babble draws at 0–5 dB.
- **Split points:** it cuts the audio in different places.

None of this costs much WER. For this model, the damage is in the recogniser, not the VAD.

## Findings

1. **Reverb dominates.** A mild room is enough: the 0.4s bucket has a measured T30 of 0.21–0.39 s,
   and the first IR drawn has 73% of its energy in the first 50 ms. With no noise at all, that room
   takes WER from 35.5% to 76% (+41 pp [+32, +49]). The 0.8s bucket gives about 94%.
2. **The model goes silent on reverb rather than guessing.** 94–98% of reverb-cell errors are
   deletions, against 57% on clean. Level is not the cause:
   - Clean turned down 8 dB, to the reverb files' RMS, gives 37.2% WER (+1.7 pp).
   - A reverb file turned up to clean level still gives 68.1%. Those are single ad-hoc runs; see
     `results_2026-09-24/level_check/` and `FAST_NOTES.md`.
3. **Noise alone degrades gracefully, but depends heavily on noise type.**
   - 20 dB is indistinguishable from clean (+0.8 pp [-1.8, +3.3]), 10 dB costs about 6 pp, and
     0 dB costs about 21 pp.
   - The spread across draws grows sharply at low SNR because babble is far worse. At 0 dB, per
     draw: restaurant 81.5%, cafeteria 66.2%, station 55.8%, office 39.6%, washing machine 37.2%.
4. **Noise on top of reverb adds comparatively little**, because the reverb has already removed
   most of the output.
5. **The implication for training.** The biggest robustness win is RIR augmentation during
   training (reverb), with babble-type noise a clear second. The current recipe only mixes MUSAN
   noise.

## External baselines: Gemini (run 2026-09-24, free tier)

`gemini_transcribe.py` sends each file to the Gemini API as a single request.
- **Audio span:** from the first to the last oracle-VAD segment, as FLAC through the Files API.
  Uploads are deleted after use. Both systems hear the same speech.
- **Normalisation:** outputs are lowercased, punctuation is stripped, hyphens are split and
  numbers are spelled out in Malay. Then they are scored exactly like our model.
- **Last column:** the paired difference against our model's oracle-VAD transcripts.
- **Evidence:** raw API responses and transcripts are in `results_2026-09-24/gemini/`.

### Gemini 3.5 Transcribe (`gemini-3.5-transcribe`)

This is a dedicated transcription model, used with no prompt. Its free tier allows 25 requests a
day and 10K tokens a minute, so it covers a 21-file subset: clean, noise-only at 10 and 0 dB,
and reverb-only in both buckets, with all 5 draws.

| RT60 bucket | SNR (dB) | WER % [95% CI] | per-draw mean ± sd | ΔWER Gemini − ours, pp [95% CI] |
|---|---|---|---|---|
| clean | clean | 37.1 [31.1, 43.3] | — | +1.6 [-4.8, +7.6] |
| none | 10 | 38.5 [32.1, 45.0] | 38.5 ± 2.8 | -3.3 [-10.4, +3.2] |
| none | 0 | 52.1 [38.8, 66.1] | 52.1 ± 15.5 | -4.0 [-9.7, +1.9] |
| 0.4s | none | 61.2 [40.2, 84.3] | 61.2 ± 27.3 | -14.8 [-33.4, +4.9] |
| 0.8s | none | 88.5 [73.4, 98.5] | 88.5 ± 16.0 | -5.1 [-19.6, +5.3] |

- **Parity.** Our model is statistically level with Gemini Transcribe in every tested cell: all
  paired CIs include 0.
- **Reverb breaks Gemini too, all or nothing.** For 3 of the 10 reverb files, it read all the
  audio (14,820 tokens) and returned an empty transcript with `finishReason: STOP`, which is 100%
  WER. Two of those three use `room_021_v4_far_4.34m`, whose reflections are 5 dB louder than the
  direct sound. On other IRs it stays near its clean level (37.6–44.5% on three of the 0.4s
  draws), whereas our model is uniformly bad (69–94%).
  - This supports finding 1: these rooms are genuinely hard, not only hard for our model.
  - Our model is the one hurt more consistently.

### Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`): not usable as an ASR baseline

This is a general LLM, prompted for a verbatim Malay transcript at temperature 0 (the prompt is
`PROMPT_MALAY`). It ran on the full grid, 71 files.

- **Clean WER:** 41.8%, which is +6.3 pp [+0.1, +12.3] against our model.
- **20 of 71 outputs broke:**
  - 12 were runaway repetition loops, up to 40,468 inserted words in one file; two hit
    `MAX_TOKENS`.
  - 7 were near-empty, with fewer than 150 words out.
  - 1 was blocked with `finishReason: RECITATION`.
- **Effect on the tables:** these failures drive cell WERs above 100%, up to 823% (full table in
  `gemini/flash_lite/result_flashlite_table.md`).
- **Even robustly it trails.** Median per-draw WER in the noise-only cells is 51–77%, against our
  36–56%.
- **Scoring rule:** digit runs longer than 12 characters are dropped as loop output before
  scoring. Nothing else is filtered.

## More baselines: Whisper, Gemini Live and Gemini Flash (run 2026-09-25)

All numbers in this section come from runs on 2026-09-25. Evidence is in
`results_2026-09-25/`: every transcript, every raw model output, and the score tables.

- **Same speech for everyone.** Whisper and Gemma decode each oracle-VAD segment separately. The
  Gemini models get the first-to-last oracle span, as in the 2026-09-24 runs.
- **Same scoring.** Outputs are normalised with `gemini_transcribe.normalise` and scored with
  `score.py`, like everything above.
- **Mostly draw 0 only.** The table below covers clean plus draw 0 of four cells
  (`results_2026-09-25/manifest_d0.csv`). With one draw per cell, the CIs cover only utterance
  resampling.
- **A caveat on the test clip.** It comes from Mesolitica, and so does the training data of both
  our model and Mesolitica's Whisper. It is home ground for both. FLEURS test, and the 3 m
  re-recording of it from Stage 3, would be a neutral test (see `exported/Stage5.tar.gz`).

### All models on the draw-0 files

WER %; ✗ means no usable output. "×2" means a second attempt with identical settings failed
the same way. "then" gives the second attempt's result where it differed.

**How each model was segmented:**
- **Oracle VAD:** the clean clip's VAD segments are reused on every file, so only the recogniser
  is tested.
- **Actual VAD:** our VAD runs on each noisy file, which is the real end-to-end pipeline.
- **Gemini models:** they get the whole file (the oracle span is 0.0–592.8 s, i.e. all of it) and
  segment it themselves, so they are end to end.
- **Whisper and Gemma:** they ran on oracle segments only.

| Model | Segmentation | clean | none, 10 dB | none, 0 dB | 0.4s, none | 0.8s, none |
|---|---|---|---|---|---|---|
| **Ours** | oracle VAD | **35.5** | 38.4 | 39.6 | 69.0 | 93.8 |
| **Ours** | actual VAD | **35.5** | 38.4 | 39.9 | 70.6 | 95.7 |
| Gemini 3.5 Transcribe (2026-09-24) | own | 37.1 | **36.2** | **36.4** | **37.6** | ✗ empty ×2 |
| Gemini 3.5 Transcribe Live | own (server VAD) | 42.4 | 43.9 | 48.3 | 51.6 | 70.8 |
| Gemini 3 Flash (`gemini-3-flash-preview`) | own | 43.3 | 44.9 | ✗ runaway ×2 | 40.8 | **42.4** |
| Gemini 3 Flash, thinking low | own | 45.7 | 47.4 | 61.4 | 42.0 | ✗ runaway |
| Gemini 3.5 Flash | own | ✗ cut short ×2 | 41.0 | ✗ empty, then ✗ runaway | 43.1 | 48.0 |
| Gemini 3.5 Flash, thinking low | own | ✗ runaway | ✗ runaway | ✗ runaway | ✗ runaway | ✗ RECITATION |
| Gemini 3.6 Flash | own | 43.4 | 45.4 | 43.8 | 47.0 | 53.6 |
| Gemini 3.8 Flash | own | 45.5 | 47.0 | 42.4 | ✗ blocked, then 46.3 | ✗ blocked ×2 |
| Gemini 3.8 Flash, thinking low | own | not run (503, then daily quota) | 45.7 | 44.8 | ✗ blocked | ✗ blocked |
| Gemma 4 E2B (local, bf16) | oracle VAD | 43.9 | 44.0 | 45.3 | 51.1 | 65.8 |
| Mesolitica Whisper turbo, standard fallback decoding | oracle VAD | 49.4 | 47.0 | 65.4 | 51.1 | 96.7 |
| Mesolitica Whisper turbo, greedy only | oracle VAD | 48.2 | 90.5 | 77.6 | 128.5 | 190.8 |
| Gemini 3.1 Flash-Lite | own | ✗ runaway | ✗ runaway | ✗ runaway | ✗ runaway | ✗ runaway |

Per-file scores for the retries, the thinking-low runs and 3.6 Flash are in
`results_2026-09-25/gemini/retry_scores.csv`, from `score_files.py`.

1. **On clean speech, nothing beats our model.** Gemini Transcribe is level with it; every other
   model is 7–14 pp worse.
2. **Reverb hurts our model far more than the others.** In the 0.8s cell the Gemini Flash models
   reach 42–48% and Transcribe Live 71%, against our 94%. Whisper's 0.8s WER (96.7%) is inflated
   by one looping segment, but it gets 538 of 1,046 reference words right there, against our 65.
   This supports finding 1: reverb is a gap in our model, not only a hard test.
3. **General-purpose LLMs are mostly unreliable transcribers.** Every Flash model except 3.6
   broke on at least one of the five files.
4. **Robust on all five files, and none of them failed:** ours, Transcribe Live, 3.6 Flash,
   Whisper with fallback and Gemma 4 E2B. On draw 0, Gemma looked the most noise-robust (clean
   43.9, 0 dB 45.3). The full grid below does not bear that out: with babble noise, Gemma loses
   ground and sometimes loops.

### Mesolitica Whisper (`mesolitica/Malaysian-whisper-large-v3-turbo-v3`)

This is Mesolitica's Malaysian fine-tune of Whisper large-v3 turbo. It is not the stock OpenAI
`large-v3` that the Stage 3 baseline used. It ran with `whisper_transcribe.py`: HF transformers,
fp32, CPU, language `ms`, no timestamps.

- **Greedy decoding loops.** Pure greedy (to match our model) falls into repetition loops on
  noisy audio: 2–5 of 42 segments per noisy file, e.g. "eh" 444 times on a 1.2 s segment. That
  adds 500–1,500 inserted words per file, hence WERs over 100%.
- **The fair baseline is the standard fallback** (`--fallback`). A repetitive or unlikely segment
  is re-decoded at temperature 0.2 … 1.0, as stock Whisper does. Some repetition survives:
  - On the 0.8s file, one segment still loops ("eh" × 384). That is 384 of the file's 1,011
    errors; without it the WER would be 59.9% rather than 96.7%.
  - On the 0 dB file, one segment is 209 repetitive words.
- **Against ours (fallback, paired Δ):**
  - Clean: +14.0 pp [+5.0, +23.2], mostly substitutions and insertions. Some of it is spelling
    style ("di kalangan" vs "dikalangan"). The reference follows Mesolitica conventions, and so
    does our training data.
  - 10 dB: +8.6 pp [−0.3, +17.6]. 0 dB: +25.8 pp [+2.2, +64.3].
  - 0.4s: −18.0 pp [−27.2, −8.4].
- **Arabic script.** On a recited Quran verse Whisper writes Arabic script, while the reference
  romanises it. That costs it 26 words on clean.
- **Speed (wall clock, 4 threads, i7-1365U).**
  - Greedy: 17–29 min per file.
  - Fallback: 22–41 min per file, but that run overlapped a Gemma job and a large scoring job.
  - Full grid (71 files): about 36 h with `--fallback` (mean about 30 min per file).
- **int8 is not a stand-in.** faster-whisper int8 (CTranslate2) matched fp32 on only 13 of 42
  clean segments, looped more (clean WER 94.6%), and was only 5% faster on clean.

### Gemini 3.5 Transcribe Live (`gemini-3.5-transcribe-live`)

This is the Live API (WebSocket) version of Transcribe, run with `gemini_live_transcribe.py`. It
has no daily request limit on the free tier. Audio is streamed at 4× real time, the server's own
VAD splits it into turns, and the final `inputTranscription` of each turn is scored.

- **Streaming pace matters for quota.** Streaming unpaced hit the per-minute token limit (socket
  close 1011).
- **Superseded.** A client bug truncated 6 of these 21 files: the stream stopped before the
  server finalised the last turn, dropping the reference's last line. The full-grid section
  below has the corrected numbers, which change these cells by at most 0.4 pp. This table is
  kept as it was.
- **Results.** It covered the same 21 files as Transcribe (clean plus five draws of four cells):

| Cell | WER % [95% CI] | Live − ours, pp | Live − Transcribe, pp |
|---|---|---|---|
| clean | 42.4 [35.5, 49.2] | +7.0 [−0.1, +13.9] | +5.4 [+1.3, +10.0] |
| none, 10 dB | 48.6 [40.8, 56.6] | +6.8 [+0.6, +12.7] | +10.1 [+5.9, +14.8] |
| none, 0 dB | 69.4 [50.9, 87.1] | +13.3 [+6.0, +20.7] | +17.3 [+9.7, +25.2] |
| 0.4s, none | 57.2 [49.7, 64.9] | −18.8 [−26.2, −11.3] | −4.0 [−27.4, +17.0] |
| 0.8s, none | 71.8 [65.1, 78.4] | −21.8 [−28.5, −15.2] | −16.8 [−29.4, −1.3] |

- **Worse on noise, better on reverb.** Live is significantly worse than both ours and Transcribe
  on noise, and significantly better than ours on reverb.
- **It never goes silent.** Transcribe emptied 3 reverb files; Live's per-draw sd on reverb is
  about 5 pp.
- **Deletions dominate its errors** (198 of 444 on clean).
- **Untested.** Whether real-time pacing, or our segmentation instead of the server VAD, would
  help.

### Gemini Flash models (prompted, `gemini_transcribe.py --prompt malay`)

Free tier, 20 requests a day each.

| Model | What broke |
|---|---|
| 3.5 Flash | On clean it stopped after about 30 s of audio (`STOP`, 36 output tokens). On 0 dB it spent 62,915 thinking tokens and returned a blank line. |
| 3 Flash | On 0 dB it spent 62,910 thinking tokens, then answered partly in French until `MAX_TOKENS`. |
| 3.8 Flash | Both reverb files were blocked: `promptFeedback.blockReason: OTHER`, no candidates. |
| 3.1 Flash-Lite | All five hit `MAX_TOKENS` with 33K–65K words each (reference: 1,046). Not scored: aligning them took over 16 GB of RAM. |
| 3.6 Flash | Nothing broke; it was the only Flash model to complete all five files. It needed several rounds of retries through HTTP 503 "high demand". |

- **Retries (same day, a second API key, identical settings).** Most failures reproduce:
  - Transcribe returned empty transcripts again on all 3 reverb files it had emptied.
  - 3 Flash hit the same runaway on 0 dB, with the identical WER (111.5).
  - 3.8 Flash was blocked again on 0.8s, but the 0.4s file went through (46.3).
  - 3.5 Flash was cut short again on clean (16 words out). On 0 dB it switched from empty to
    runaway (248.4%).
- **Lowering thinking does not fix them.** `--thinking low` helped 3 Flash on 0 dB (61.4, from a
  runaway) but broke its 0.8s file into a runaway. It made 3.5 Flash loop on every file
  (27K–65K words out), and the fifth was blocked for `RECITATION`.
- **These models are not usable baselines.** Failures that reproduce with the same settings are
  properties of the model on that audio, not bad luck.
- **Unavailable models.** `gemini-2.5-flash` is no longer offered to new keys (HTTP 404). The
  hosted Gemma 4 models (`gemma-4-31b-it`, `gemma-4-26b-a4b-it`) reject audio.

### Gemma 4 E2B (`google/gemma-4-E2B-it`, local)

Only the E2B, E4B and 12B checkpoints take audio, and only in clips of up to 30 s.
`gemma_transcribe.py` runs E2B in bfloat16 on the CPU with the model card's ASR prompt.

- **Speed.** 89–106 minutes per 532.6 s of segment audio, roughly 10–12× slower than real time,
  with 4 threads.
- **Results** (`results_2026-09-25/gemma/`, paired Δ against our oracle-VAD run):

| Cell | WER % [95% CI] | Δ vs its clean, pp | Gemma − ours, pp |
|---|---|---|---|
| clean | 43.9 [37.0, 51.0] | — | +8.4 [+1.6, +14.8] |
| none, 10 dB | 44.0 [36.8, 51.4] | +0.1 [−2.2, +2.6] | +5.5 [−1.9, +12.6] |
| none, 0 dB | 45.3 [38.3, 52.8] | +1.4 [−1.3, +4.5] | +5.7 [−1.6, +12.6] |
| 0.4s, none | 51.1 [44.0, 58.4] | +7.2 [+3.6, +10.9] | −18.0 [−25.9, −10.2] |
| 0.8s, none | 65.8 [59.4, 72.5] | +21.9 [+17.5, +26.4] | −28.0 [−34.3, −21.3] |

- **Noise barely moves it.** Office noise at 0 dB costs 1.4 pp. That is draw 0 only; babble was
  the hard noise type for our model and has not been tried.
- **Reverb costs it much less than it costs ours.**
- **No failures on these five files.** It had no runaway segments and one empty segment. The
  full grid (below) does have loops.

## Full-grid baselines (weekend run from 2026-09-25)

These are the worthwhile baselines on all 71 files (15 cells × 5 draws, plus clean), scored
with `score.py` against our oracle-VAD run. The run started 2026-09-25 23:37; each model's
completion date is given with its table. Evidence is in `results_grid/`, and ids were run
draw-major (`results_grid/ids_by_draw.txt`).

## Full-grid comparison

**This is the main results table.** Every system is scored on all 71 files: clean, plus 14 cells ×
5 draws. Completed 2026-09-28.

- Each cell is WER % [95% CI] from `score.py`'s paired bootstrap.
- ▼ / ▲: significantly better / worse than our model with oracle VAD (the paired 95% CI of the
  difference excludes 0).
- **Bold:** lowest WER in the row.
- For our noisy-VAD column, the marker is the VAD cost.
- Generated by `results_grid/make_comparison.py` from the tracked cells CSVs; saved as
  `results_grid/comparison.md`.
- Segmentation:
  - Our model, Gemma and Whisper decode the clean-audio (oracle) VAD segments.
  - The Gemini systems get the whole file and segment it themselves.
  - "Ours, noisy VAD" is our real end-to-end pipeline.

| RT60 bucket | SNR (dB) | Ours, oracle VAD | Ours, noisy VAD | Gemini Transcribe | Gemini Transcribe Live | Gemma 4 E2B | Whisper turbo + fallback |
|---|---|---|---|---|---|---|---|
| clean | clean | **35.5 [29.0, 42.3]** | **35.5 [29.0, 42.3]** | 37.1 [31.1, 43.3] | 42.4 [35.5, 49.2] | 43.2 [36.4, 50.2] ▲ | 49.4 [40.0, 59.9] ▲ |
| none | 20 | 36.3 [30.0, 42.9] | **35.8 [29.6, 42.4]** | 38.6 [31.9, 45.4] | 42.2 [35.8, 48.8] | 44.3 [37.2, 51.6] ▲ | 47.5 [38.7, 57.0] ▲ |
| none | 10 | 41.8 [33.9, 50.1] | 40.7 [33.1, 48.6] | **38.5 [32.1, 45.1]** | 48.3 [40.4, 56.4] ▲ | 47.2 [39.6, 55.1] | 51.8 [41.2, 66.4] |
| none | 5 | 47.6 [37.4, 58.5] | 46.9 [36.2, 58.5] | **42.9 [35.4, 51.0]** | 56.4 [44.2, 69.6] ▲ | 51.0 [42.2, 60.3] | 52.3 [43.1, 61.9] |
| none | 0 | 56.0 [40.6, 72.2] | 56.4 [40.4, 73.0] | **52.1 [38.8, 66.1]** | 69.0 [50.2, 87.0] ▲ | 59.0 [46.1, 71.9] | 61.6 [48.4, 76.2] |
| 0.4s | none | 76.0 [66.9, 84.2] | 78.3 [69.2, 86.3] ▲ | 61.2 [40.0, 84.3] | 57.0 [49.3, 64.5] ▼ | 59.4 [48.5, 77.2] | **55.9 [44.2, 78.0]** |
| 0.4s | 20 | 77.1 [68.0, 85.3] | 78.8 [69.5, 87.0] | **54.4 [43.7, 65.0]** ▼ | 67.5 [56.4, 77.6] ▼ | 58.5 [50.1, 67.0] ▼ | 58.2 [45.5, 81.7] |
| 0.4s | 10 | 81.4 [71.7, 89.4] | 83.1 [72.6, 91.6] | **58.8 [41.3, 78.5]** ▼ | 79.5 [65.6, 92.2] | 65.4 [54.2, 76.3] ▼ | 65.1 [50.8, 94.3] |
| 0.4s | 5 | 84.7 [73.6, 94.0] | 86.5 [75.0, 96.0] | 80.5 [59.7, 98.6] | 83.7 [69.8, 96.6] | 75.1 [57.7, 98.9] | **66.8 [54.3, 78.1]** ▼ |
| 0.4s | 0 | 87.1 [75.6, 96.8] | 87.4 [75.2, 97.9] | **79.8 [56.8, 99.9]** | 88.5 [76.0, 99.5] | 92.2 [60.3, 147.2] | 80.4 [59.1, 103.0] |
| 0.8s | none | 93.6 [90.6, 96.2] | 95.2 [92.5, 97.3] | 88.5 [73.6, 98.5] | 71.8 [65.0, 78.4] ▼ | **66.7 [59.4, 74.0]** ▼ | 76.9 [58.0, 110.0] |
| 0.8s | 20 | 93.8 [90.9, 96.3] | 94.8 [92.3, 96.9] | 74.6 [55.2, 92.5] ▼ | 85.3 [78.2, 91.4] ▼ | **69.1 [61.6, 76.2]** ▼ | 77.3 [60.1, 114.5] |
| 0.8s | 10 | 95.0 [91.9, 97.6] | 95.9 [93.6, 97.7] | 80.6 [61.6, 98.0] | 91.3 [84.4, 97.3] | **79.7 [68.4, 96.5]** | 86.1 [67.5, 121.3] |
| 0.8s | 5 | 95.6 [92.5, 97.9] | 96.0 [92.6, 98.5] | 83.8 [68.4, 95.5] ▼ | 94.0 [88.7, 98.5] | 87.0 [74.3, 101.0] | **80.0 [68.4, 91.2]** ▼ |
| 0.8s | 0 | 96.6 [93.5, 99.0] | 96.7 [93.4, 99.0] | 88.3 [76.2, 98.6] ▼ | 96.7 [91.8, 99.7] | 89.5 [78.0, 100.3] | **87.6 [76.7, 96.6]** ▼ |

What the grid shows:

1. **Clean and light noise (20 dB):** our model is best. Gemini Transcribe is level with it,
   and Gemma, Whisper and Live are worse (significantly so for Gemma and Whisper).
2. **Heavier noise (10 to 0 dB):** Gemini Transcribe has the lowest WER, but no system is
   significantly better than ours. Live is significantly worse.
3. **Reverb:** every other system beats ours in some reverb cells. None is significantly worse
   than ours in any reverb cell. Significant wins over ours, out of 10 reverb cells:
   - Transcribe: 5;
   - Transcribe Live: 4;
   - Gemma: 4;
   - Whisper: 3.

   Our model's weakness is reverb. That supports finding 1 and the training recommendation:
   RIR augmentation first.
4. **Reverb plus heavy noise (0.4s or 0.8s at 0–5 dB) defeats everyone.** The best result in
   those four cells is 66.8% (Whisper, 0.4s at 5 dB).
5. **Failure modes behind the wide CIs:**
   - Transcribe returns empty transcripts on 12 of 50 reverb files.
   - Whisper loops on 18 of 71 files, even with fallback.
   - Gemma loops on 9 of 71.
   - Live and our model are the most predictable.
6. **Our VAD costs almost nothing.** Noisy vs oracle VAD differs by at most 2.2 pp, and
   significantly only in the 0.4s no-noise cell.

### Gemini 3.5 Transcribe Live (complete 2026-09-26 02:05)

`--speed 4`. The 21 files from the subset run were copied in, not re-run, so on those files
this table matches the subset table by construction.

| RT60 bucket | SNR (dB) | WER % [95% CI] | Live − ours, pp [95% CI] |
|---|---|---|---|
| clean | clean | 42.4 [35.5, 49.2] | +7.0 [−0.1, +13.9] |
| none | 20 | 42.2 [35.8, 48.8] | +5.9 [−1.0, +12.4] |
| none | 10 | 48.3 [40.4, 56.4] | +6.5 [+0.2, +12.5] |
| none | 5 | 56.4 [44.2, 69.6] | +8.8 [+2.3, +15.2] |
| none | 0 | 69.0 [50.2, 87.0] | +13.0 [+5.4, +20.5] |
| 0.4s | none | 57.0 [49.3, 64.5] | −19.0 [−26.5, −11.7] |
| 0.4s | 20 | 67.5 [56.4, 77.6] | −9.6 [−18.0, −0.7] |
| 0.4s | 10 | 79.5 [65.6, 92.2] | −1.8 [−10.2, +6.0] |
| 0.4s | 5 | 83.7 [69.8, 96.6] | −1.0 [−7.4, +4.0] |
| 0.4s | 0 | 88.5 [76.0, 99.5] | +1.4 [−3.5, +5.2] |
| 0.8s | none | 71.8 [65.0, 78.4] | −21.8 [−28.8, −15.1] |
| 0.8s | 20 | 85.3 [78.2, 91.4] | −8.5 [−15.4, −2.6] |
| 0.8s | 10 | 91.3 [84.4, 97.3] | −3.7 [−9.7, +0.9] |
| 0.8s | 5 | 94.0 [88.7, 98.5] | −1.6 [−5.7, +1.4] |
| 0.8s | 0 | 96.7 [91.8, 99.7] | +0.1 [−3.5, +2.9] |

- **Truncation fix (2026-09-26).** The review found that 8 of 71 streams ended before the
  server finalised a last turn that ran to the end of the audio, so that text was lost.
  - The client now ends each stream with 2 s of silence.
  - Those 8 files were re-run, and none is truncated now (`--rebuild` checks every log). The old
    logs are kept as `raw/<id>.truncated.jsonl`.
  - This table uses the re-run files. The correction lowers WER by at most 0.4 pp per cell.
- **Noise:** Live is worse than ours at every SNR, and the gap grows as SNR falls (+6 to +13 pp).
- **Reverb:** its advantage holds only with little or no noise (none or 20 dB). With reverb plus
  noise at 10 dB or louder it is level with ours. The combined condition breaks both models.
- **Near-empty outputs:** a few files returned almost nothing, e.g. `rt0.8_snr0_d1`: 2 words.
  Most combine 0 dB restaurant or cafeteria babble with reverb. The exceptions are
  `rt0.8_snr5_d1` (14 words, 5 dB) and `rt0.8_snr0_d3` (35 words, station noise). They are real
  outputs, not failed requests.
- **Stream retries.** None of the 50 new files needed one. Earlier, in the subset run,
  `rtnone_snr10_d2` lost its socket twice before succeeding on the third attempt; both failed
  logs are kept as `raw/*.fail<N>.jsonl`.

### Gemini 3.5 Transcribe (complete 2026-09-26 03:07)

One request per file, no prompt. The 21 files from 2026-09-24 were reused unchanged; the other
50 were run on 2026-09-26 over two API keys.

| RT60 bucket | SNR (dB) | WER % [95% CI] | Transcribe − ours, pp [95% CI] | Transcribe − Live, pp [95% CI] |
|---|---|---|---|---|
| clean | clean | 37.1 [31.1, 43.3] | +1.6 [−4.8, +7.6] | −5.4 [−10.0, −1.3] |
| none | 20 | 38.6 [31.9, 45.4] | +2.3 [−4.8, +9.2] | −3.6 [−7.7, +0.8] |
| none | 10 | 38.5 [32.1, 45.1] | −3.3 [−10.3, +3.2] | −9.8 [−14.6, −5.6] |
| none | 5 | 42.9 [35.4, 51.0] | −4.7 [−12.3, +2.5] | −13.5 [−21.5, −6.1] |
| none | 0 | 52.1 [38.8, 66.1] | −4.0 [−9.7, +1.9] | −16.9 [−25.0, −9.2] |
| 0.4s | none | 61.2 [40.0, 84.3] | −14.8 [−33.7, +4.7] | +4.2 [−17.1, +27.3] |
| 0.4s | 20 | 54.4 [43.7, 65.0] | −22.7 [−32.5, −13.0] | −13.1 [−24.3, −1.1] |
| 0.4s | 10 | 58.8 [41.3, 78.5] | −22.6 [−34.9, −8.2] | −20.8 [−31.7, −7.8] |
| 0.4s | 5 | 80.5 [59.7, 98.6] | −4.2 [−17.5, +5.7] | −3.2 [−13.5, +3.8] |
| 0.4s | 0 | 79.8 [56.8, 99.9] | −7.3 [−21.2, +4.1] | −8.7 [−20.7, +0.8] |
| 0.8s | none | 88.5 [73.6, 98.5] | −5.1 [−19.5, +5.3] | +16.8 [+1.7, +29.6] |
| 0.8s | 20 | 74.6 [55.2, 92.5] | −19.2 [−38.1, −1.8] | −10.7 [−27.9, +5.6] |
| 0.8s | 10 | 80.6 [61.6, 98.0] | −14.4 [−32.7, +2.0] | −10.7 [−27.1, +3.8] |
| 0.8s | 5 | 83.8 [68.4, 95.5] | −11.8 [−25.9, −0.7] | −10.2 [−23.7, +1.0] |
| 0.8s | 0 | 88.3 [76.2, 98.6] | −8.3 [−18.4, −0.1] | −8.4 [−19.2, −0.8] |

- **Noise only:** statistically level with ours at every SNR (all CIs include 0).
- **Reverb:** better than ours in every reverb cell. The difference is significant in 5 of 10
  (0.4s at 20 and 10 dB; 0.8s at 20, 5 and 0 dB).
- **Against Live:** significantly better in 7 of 15 cells: clean, every noise-only cell below
  20 dB, 0.4s at 20 and 10 dB, and 0.8s at 0 dB. It is level in 7, and significantly worse only
  in the pure 0.8s cell, where its empty outputs cost it.
- **12 of 71 files came back empty**, which scores 100% WER. They follow a clear pattern:
  - All 12 are reverb files (24% of the 50 reverb files). None of the noise-only files is
    empty.
  - 10 of the 12 are draws 1 and 2: restaurant and cafeteria babble, plus those draws' IRs.
  - The retries on 2026-09-25 showed these failures reproduce.
- **Its reverb CIs are wide** because of this all-or-nothing behaviour. Where it answers, it
  is much better than ours.

### Gemma 4 E2B, fp32 (complete 2026-09-27 09:38)

Oracle segments, model-card prompt, greedy, fp32. It took about 34 h for 71 files, about 29 min
each, with no errors or restarts.

| RT60 bucket | SNR (dB) | WER % [95% CI] | per-draw mean ± sd | Gemma − ours, pp [95% CI] | Gemma − Transcribe, pp [95% CI] |
|---|---|---|---|---|---|
| clean | clean | 43.2 [36.4, 50.2] | — | +7.7 [+1.1, +14.0] | +6.1 [+2.8, +9.6] |
| none | 20 | 44.3 [37.2, 51.6] | 44.3 ± 0.8 | +8.0 [+1.4, +14.2] | +5.7 [+0.9, +10.0] |
| none | 10 | 47.2 [39.6, 55.1] | 47.2 ± 3.8 | +5.4 [−1.5, +12.0] | +8.7 [+5.3, +12.4] |
| none | 5 | 51.0 [42.2, 60.3] | 51.0 ± 7.6 | +3.4 [−3.5, +10.1] | +8.1 [+4.6, +12.0] |
| none | 0 | 59.0 [46.1, 71.9] | 59.0 ± 14.5 | +3.0 [−3.3, +9.7] | +7.0 [+3.4, +10.7] |
| 0.4s | none | 59.4 [48.5, 77.2] | 59.4 ± 9.1 | −16.6 [−28.1, +0.0] | −1.8 [−27.7, +24.9] |
| 0.4s | 20 | 58.5 [50.1, 67.0] | 58.5 ± 4.0 | −18.6 [−26.6, −10.2] | +4.1 [−5.5, +13.8] |
| 0.4s | 10 | 65.4 [54.2, 76.3] | 65.4 ± 11.0 | −15.9 [−22.9, −8.9] | +6.7 [−7.6, +17.0] |
| 0.4s | 5 | 75.1 [57.7, 98.9] | 75.1 ± 20.5 | −9.6 [−19.5, +9.7] | −5.4 [−17.3, +11.8] |
| 0.4s | 0 | 92.2 [60.3, 147.2] | 92.2 ± 45.8 | +5.2 [−17.7, +54.7] | +12.4 [−8.7, +58.4] |
| 0.8s | none | 66.7 [59.4, 74.0] | 66.7 ± 4.6 | −26.9 [−33.9, −19.8] | −21.8 [−33.2, −9.7] |
| 0.8s | 20 | 69.1 [61.6, 76.2] | 69.1 ± 6.2 | −24.7 [−31.9, −17.8] | −5.5 [−20.1, +9.2] |
| 0.8s | 10 | 79.7 [68.4, 96.5] | 79.7 ± 13.3 | −15.4 [−25.5, +0.5] | −0.9 [−14.2, +13.8] |
| 0.8s | 5 | 87.0 [74.3, 101.0] | 87.0 ± 13.4 | −8.6 [−19.9, +4.2] | +3.2 [−9.2, +16.4] |
| 0.8s | 0 | 89.5 [78.0, 100.3] | 89.5 ± 11.7 | −7.1 [−16.8, +2.6] | +1.2 [−5.0, +9.4] |

- **Noise:** Gemma is worse than ours on clean and at 20 dB, and level with ours from 10 to 0 dB.
  It is significantly worse than Transcribe at every noise-only level.
  - The draw-0 impression of noise robustness came from office noise. Babble (draws 1–2) costs
    it much more: the 0 dB per-draw sd is 14.5 pp.
- **Reverb:** it beats ours in 4 of 10 reverb cells: 0.4s at 20 and 10 dB, 0.8s with no noise
  and at 20 dB. It is level with ours in the other 6.
  - Against Transcribe it is level in every reverb cell but one. It wins pure 0.8s, where
    Transcribe returns empty transcripts.
- **Loops:** 9 of 71 files contain segments that loop up to the 256-token cap, e.g. "thì thì
  thì …", drifting into Vietnamese.
  - All 9 are reverb files, and 7 of them are babble draws (1–2).
  - The cap bounds the damage, but it drives the 0.4s / 0 dB cell to 92% [60, 147].
  - Listed in the 2026-09-27 check; the raw text is in `results_grid/gemma_fp32/raw/`.
- **fp32 vs bf16:** the 5 draw-0 files, run in both dtypes, match on only 131 of 210 segments.
  The accuracy is close (clean 43.2 vs 43.9), but this grid and the bf16 draw-0 table above are
  different runs.

### Mesolitica Whisper turbo + fallback (complete 2026-09-28 18:53)

Oracle segments, HF transformers fp32, `--fallback`. The 5 draw-0 files are from 2026-09-25; the
other 66 ran 2026-09-27/28 at about 28 min each, with no errors or restarts. Paired differences
are in `results_grid/result_whisper_vs_{oracle,transcribe,gemma}_*`.

- **Against ours:**
  - significantly worse on clean (+14.0 pp) and at 20 dB (+11.2);
  - level from 10 to 0 dB;
  - significantly better in 3 reverb cells: 0.4s at 5 dB, and 0.8s at 5 and 0 dB.
- **Against Transcribe:** significantly worse on clean and at every noise-only SNR; level in
  every reverb cell.
- **Against Gemma:** significantly worse on clean (+6.2 pp); level everywhere else.
- **Loops:** 18 of 71 files contain a segment that still loops after the fallback (> 6 words
  per second of audio).
  - Almost all are reverb files, 12 of them in the 0.8s bucket, plus 2 noise-only.
  - That is why many of its reverb CIs reach above 100%.

### Abandoned: Gemini 3.6 Flash and 3.8 Flash (2026-09-26)

Both were started on the full grid and stopped at the user's request. Their partial outputs and
per-file scores are kept in `results_grid/flash36/`, `results_grid/flash38/` and
`results_grid/abandoned_flash_scores.csv`.

- **3.6 Flash: stopped at 6 of 71.**
  - HTTP 503 "high demand" rejections still count towards the 20-a-day quota. On 2026-09-26
    they used up both keys' quota after one new file.
  - No file failed. Its 5-file draw-0 results above stand.
- **3.8 Flash: stopped at 22 of 71, because it is unusable as a baseline.**
  - 12 of the 22 outputs were blocked or empty (`promptFeedback.blockReason: OTHER`).
  - Blocks cover reverb in both buckets, including the 0.4s file with no noise. They also hit
    restaurant babble with no reverb (`rtnone_snr5_d1`, `rtnone_snr0_d1`).
  - Where it answers, its median per-file WER is 46.7%.

## Bugs found and fixed along the way

- **`predict.py`: crash on very short segments.** Noisy audio makes the VAD emit slivers too short
  to decode, and the model crashed on them. It now skips a segment only if it would have crashed:
  under 400 samples, or under 9 fbank frames.
- **`predict.py`: long-segment splitting could recurse forever.**
  - The split-point search was recursive and hit Python's recursion limit. It is now an iterative
    loop with the same visiting order.
  - The boundary fallback computed the midpoint but never assigned it, so a split could make no
    progress. The assignment is added.
  - Both cases only triggered at 0 dB restaurant noise, where the VAD marks about 591 of 592 s
    as speech.
- **`predict.py` refactor.** `infer()` is split into `load_wav()`, `segment()` and `decode()`, so
  segments can be reused (the oracle VAD).
- **Outputs unchanged.** Clean output is byte-identical to the original GPU run
  (`predict_mesolitica.txt`), both from the CLI and from `transcribe_many.py`. Noisy transcripts
  from before the refactor also re-ran byte-identical.
- **`wer.py`: leading deletions went uncounted.**
  - The alignment never set the traceback op for column 0, so when a hypothesis missed the
    opening words the traceback stopped early. Those reference words were then left out of both
    the error count and the denominator, which made WER optimistic.
  - It affected 52 of 70 noisy-VAD runs, by 1–283 words, and 53 of 70 oracle runs, by up to 921
    words.
  - The clean result is unaffected: the stat file is byte-identical to the original.

## Evidence and checks

`results_2026-09-24/` holds the raw evidence as well as the summaries:
- `manifest.csv`: every run's cell, draw, IR (with measured T30), noise clip and slice seed.
- `transcripts/{noisy_vad,oracle_vad}/`: every transcript (`<id>.txt`) and the segments it was
  decoded with (`<id>.segments.json`), 71 runs each including `clean`. The oracle segments are in
  `oracle_vad/oracle.segments.json`.
- `result_*_cells.csv`, `result_*_runs.csv`, `result_*_table.md` and `result_vadstats.md`: the
  tables above.
- `level_check/`: the loudness test from finding 2.

The tables regenerate from these files without any audio. For example, the command below
reproduces `result_oracle_cells.csv` exactly (verified 2026-09-24):

```bash
$PY Stage_5_Eval/noise_eval/score.py $D/manifest.csv $D/transcripts/oracle_vad $GT out   # D=results dir
```

Checks run during this work:
- **Augmentation, draw 0:** output lengths equal the input, measured SNR is within 0.01 dB of the
  target, the lag to dry is 16 and 144 samples, and there is no clipping. Room reverb correlates
  1.0000 with a direct `fftconvolve`.
- **Byte-identity:** the clean output is byte-identical to the original GPU run, via both the
  `predict.py` CLI and `transcribe_many.py`, after every `predict.py` change. Oracle mode applied
  to the clean clip also reproduces it.
- **`wer.py` fix:** the clean stat file is byte-identical to the original `stat_mesolitica.txt`.
- **`wer_utt.py` guard:** its assert (per-utterance sums equal `wer.py`'s totals) was sabotaged
  by dropping insertion attribution, and it failed as expected (20 insertions against 43).

## Reproduce

Paths are relative to the repo root. Use Git Bash, and set `OMP_NUM_THREADS=4` to leave CPU for
other work.

```bash
FAST_PY=C:/Users/user/anaconda3/envs/farfield-audio-synthesis-toolkit/python.exe
PY=C:/Users/user/anaconda3/envs/collaboration-stt/python.exe
E=.scratch/noise_eval
CW=.scratch/meso/Stage_5_Eval/input/mesolitica.wav      # from exported/Stage5.tar.gz
GT=.scratch/meso/Stage_5_Eval/input/gt_mesolitica.txt

$FAST_PY Stage_5_Eval/noise_eval/augment.py $CW $E/wav --draws 5                  # ~2 min, 1.3 GB
$PY Stage_5_Eval/noise_eval/transcribe_many.py $E/wav/manifest.csv $E/pred        # ~40 s/file
$PY Stage_5_Eval/noise_eval/transcribe_many.py $E/wav/manifest.csv $E/pred_oracle --oracle-segments $CW
cp $E/pred/clean.txt $E/pred_oracle/    # the clean transcript; score.py expects it in both dirs
                                        # (produce it by running transcribe_many.py on a one-row manifest for $CW)
$PY Stage_5_Eval/noise_eval/score.py $E/wav/manifest.csv $E/pred_oracle $GT $E/result_oracle
$PY Stage_5_Eval/noise_eval/score.py $E/wav/manifest.csv $E/pred $GT $E/result_vad --vs $E/pred_oracle
$PY Stage_5_Eval/noise_eval/vad_stats.py $E/wav/manifest.csv $E/pred $E/pred_oracle/oracle.segments.json $E/result_vadstats.md
```

`augment.py` seeds everything with `zlib.crc32`, so rerunning it reproduces the same draws.

Gemini baselines (needs the API key in `~/.gemini_api_key`; free-tier quotas as noted above):

```bash
M=$E/manifest_with_clean.csv   # wav/manifest.csv plus a "clean" row pointing at $CW
S=Stage_5_Eval/noise_eval/results_2026-09-24/oracle.segments.json
$PY Stage_5_Eval/noise_eval/gemini_transcribe.py $M $S $E/gemini_transcribe --ids <subset>        # 3.5 Transcribe
$PY Stage_5_Eval/noise_eval/gemini_transcribe.py $M $S $E/gemini_flashlite \
    --model gemini-3.5-flash-lite --prompt malay --min-interval 5
$PY Stage_5_Eval/noise_eval/score.py <manifest> $E/gemini_transcribe $GT out --vs <ours oracle_vad dir>
```

2026-09-25 baselines (Whisper and Gemma need `pip install transformers`; Gemma also needs `pillow`
and `torchvision==0.23.0` from the PyTorch CPU index, which keeps torch at 2.8):

```bash
R=Stage_5_Eval/noise_eval/results_2026-09-25
D0=clean,rtnone_snr10_d0,rtnone_snr0_d0,rt0.4_snrnone_d0,rt0.8_snrnone_d0
$PY Stage_5_Eval/noise_eval/whisper_transcribe.py $M $S $E/whisper_fb --fallback --ids $D0       # ~30 min/file
$PY Stage_5_Eval/noise_eval/gemini_live_transcribe.py $M $S $E/live --ids <subset> --speed 4     # ~3 min/file
$PY Stage_5_Eval/noise_eval/gemini_transcribe.py $M $S $E/flash38 --model gemini-3.8-flash \
    --prompt malay --min-interval 15 --ids $D0
$PY Stage_5_Eval/noise_eval/gemma_transcribe.py $M $S $E/gemma --ids clean                      # ~2 h/file
$PY Stage_5_Eval/noise_eval/score.py $R/manifest_d0.csv $R/whisper/fallback $GT out \
    --vs Stage_5_Eval/noise_eval/results_2026-09-24/transcripts/oracle_vad
```

The last command reproduces `results_2026-09-25/whisper/result_fallback_cells.csv` byte for byte,
and the same command on `gemini/live` with the Transcribe subset manifest reproduces
`result_live_vs_ours_cells.csv` (both verified 2026-09-25).
