# Gemma 4 12B QAT (q4_0) via llama.cpp: feasibility probe, 2026-09-29

HANDOFF open item 1, step 1. Three clean oracle segments (0-2 of `mesolitica.wav`, the same cuts as
`results_grid/gemma_fp32/raw/clean.json`) were run through
`google/gemma-4-12B-it-qat-q4_0-gguf` (model plus `mmproj-…gguf`) on llama.cpp `b11205`,
the win-cpu-x64 build, with 8 threads on the i7-1365U.

## What runs

- **`llama-mtmd-cli` aborts with `0xC0000409`** (fail-fast) right after `init_audio`, with or
  without `--audio`. It even aborts on a text-only prompt, so the fault is in the CLI after the
  projector loads, not in audio decoding. `mtmd_cli_crash_excerpt.log` has the relevant lines.
- **Text-only works:** `llama-completion` without the projector decodes at ~3.3 tok/s.
- **`llama-server` with the projector works.** Audio goes in as OpenAI `input_audio`, with greedy
  decoding and `enable_thinking: false`. `probe.py` is the client; it read its prompt from a
  scratch file holding `gemma_transcribe.py`'s `PROMPT`.
- **The projector** is `gemma4uv`. Its audio side is a single `mm.a.input_projection` (640 → 3840)
  fed with raw 40 ms frames. There is no mel front end (`n_fft = -1`), and llama.cpp flags it
  "experimental".

## Cost

- Prompt processing runs at about 9 tok/s and generation at about 3.6 tok/s.
- **A 15 s segment takes 90–120 s.** That extrapolates to roughly 1 h per 592 s file, so about
  70 h for the full grid.

## Quality: poor on this clip

`raw/*.json` holds the full server responses. The E2B fp32 comparison is from
`results_grid/gemma_fp32/raw/clean.json`.

| Seg | Gemma 4 E2B fp32 (transformers) | Gemma 4 12B QAT (llama.cpp) |
|---|---|---|
| 0 | "aja di Mesir ketika itu qala ja'alni ala khaza inil ardi inni hafizun alim…" | The Quranic verse in Arabic script (correct content, wrong script for the reference), plus an unrequested "English:" translation |
| 1 | "So ada satu hari ni adalah seorang uncle ni dia datang beli makanan…" | "So, I just got to Harini, I'll last to Ora Uncle, you that tank, belly makan…", plus a translation |
| 2 | "Okey dia tak semua kita yang ada masalah lari…" | "Okay, bitas semu keteyar di masa lalu lagi keren…", plus a translation |
| 1, Malay-primed prompt | — | Loops "I don't know, I don't know…" until `max_tokens` (`raw/clean_1_ms.json`) |

On segments 1 and 2 the 12B transcribes Malay phonetically, as English-like gibberish, where E2B
is close to correct. It also ignores the instruction to output only the transcription.

## Open question: the model, or the port and quantisation?

The probe cannot tell which is at fault. Upstream issue ggml-org/llama.cpp#24138, "Gemma 4 12B
audio processing not working", reports looping and gibberish on 12B audio while E4B works. Some
reporters say short clips work with thinking off, and one reports the same loop in transformers
bf16. The issue was closed as stale, not fixed.

The deciding test is 12B bf16 in transformers on segment 1. Its cost:
- a download of about 24 GB, which is not yet in the HF cache;
- about 24 GB of RAM;
- about 15 minutes of compute.
