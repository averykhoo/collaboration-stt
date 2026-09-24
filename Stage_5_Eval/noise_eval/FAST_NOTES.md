# Driving the Far-field Audio Synthesis Toolkit (FAST) from this repo

These notes record how `augment.py` uses FAST and the traps found while setting it up. FAST lives
in the sibling repo `../Farfield-Audio-Synthesis-Toolkit` at commit `b5bab27`, and runs in its own
conda env, `farfield-audio-synthesis-toolkit` (Python 3.10). It was investigated on 2026-09-24.

Each item carries a label:
- **verified** means checked first-hand, by measuring output or reading the code.
- **per probe** means it comes from a read-only survey of the FAST code and was not re-checked.

## Which template

- **There is no template literally called "noise template".** The full chain is
  `src/workflow/templates.py::FASTSynthesisTemplate`, stages I–VIII (verified: it is the one used).
  - Stage I: tempo/pitch via rubberband.
  - Stage II: noise beds, optionally with their own room IR.
  - Stage III: occlusion/device IRs.
  - Stage IV: room IR.
  - Stage V: SNR mix.
  - Stage VI: joint peak normalisation.
  - Stage VII: Opus codec.
  - Stage VIII: phone lowpass.
- **Setting a stage's entry key to `None` switches it off.** The `add_room_reverb` and
  `stationary_nonstationary_NNR` keys must still be present (per probe).
- **Related templates** (per probe), both in `src/workflow/experiments.py`:
  - `OpusAblationTemplate`: noise only, and always adds Opus.
  - `FASTOpusAblationTemplate`: the full chain followed by an Opus grid.

## How `augment.py` drives it

- **Build the params `entry` by hand, not with FAST's sampler.** The sampler
  (`src/generation/generate_parameters_json.py` with `config/*.yaml`) draws tempo from [0.8, 1.2]
  and pitch from [−3, 3] (verified in `config/sample_config.yaml`). Building the entry directly
  fixes every value exactly.
- **Assemble the graph and pull only the dirty output:**
  `ch = t._assemble(); ExecutionGraph(leaves=[ch.dirty]).execute(ctx)[ch.dirty].waveform`.
  Then write it with `soundfile` as 16-bit PCM. The clean sink never runs.
- **Node paths must be relative to `ExecutionContext.input_dir`/`output_dir`.** Absolute paths
  are rejected (verified: `src/workflow/models/nodes/io.py::_portable_relative_path`).
  `augment.py` roots the context at the common parent of both repos and the output folder.
- **Resolve CLI paths before `os.chdir(FAST)`.** `augment.py` changes into the FAST root to
  import `src.*`, so relative paths must become absolute first. This was a real bug on the first
  run.
- **Seeding.** Seeds come from `zlib.crc32`, not `hash()`, which is salted per process for
  strings. Draw *d* reuses the same noise clip and slice seed in every cell, and the same IR at
  every SNR within a bucket.

## Behaviour to rely on or watch

- **Output length and alignment** (verified on the 592 s clip): the output length exactly equals
  the input length. The lag to the dry signal was 16 samples for a 0.4s-bucket IR and 144 samples
  for a 0.8s-bucket IR.
  - With a device IR switched on, output grows by `len(IR) − 1` samples (per probe).
- **Room reverb is exact convolution.** FAST's output correlates 1.0000 with
  `scipy.signal.fftconvolve(dry, ir)` once the IR's onset is trimmed (verified).
  - The onset trim is `primitives/alignment.py::align_by_room_ir_onset`.
  - Comparing without that alignment gives misleading negative correlations.
- **SNR is energy-based over the region where noise is non-zero,** against the *reverberant*
  speech, including speech pauses. There is no speech VAD (verified by name:
  `primitives/mixing.py::active_region_noise_scale`). For a looped stationary bed, the measured
  SNR matched the target within 0.01 dB (verified).
  - **Non-stationary noise is a single zero-padded event,** so its SNR only covers the event's
    region, and most of a long file stays clean (per probe). Use stationary noise for long clips.
- **Joint normalisation scales the dirty output to a 0.99 peak** and applies the same factor to
  clean (verified: `src/utils/constants.py::DEFAULT_PEAK_LEVEL`).
  - Absolute level is therefore not preserved: the reverb files came out about 8 dB quieter than
    the dry clip.
  - For the Stage 5 model that level change is worth only about 1.7 pp of WER (see `RESULTS.md`).
- **Rubberband ≥ 4 caps a single process call at 524,288 samples** (about 32.8 s at 16 kHz)
  (verified: comment in `src/audio_effects/tempochange_pitchshift_lowpass.py`).
  - Any tempo ≠ 1.0 or pitch ≠ 0.0 would both time-stretch and hit this cap on a 10-minute file.
  - Rubberband is bypassed only at exactly 1.0 / 0.0.
- **RT60 comes from the IR folder (bucket), not from a synthesised room.**
  - The folder name is an upper bound on the measured T30, which comes from
    `room_IRs/rir_metadata_audit.csv`, column `actual_physical_rt60`.
  - The `t60_X` in IR filenames is the simulation target and is mostly flagged anomalous. Ignore
    it.
  - The 0.8s bucket has only 22 IRs, so draws collide easily: 2 of 5 did in this run.

## Assets

All asset paths are under `data/input/`, and all of them are real files, not LFS pointers (per
probe; the IRs and noise used here loaded fine).

| Asset | Path | Count |
|---|---|---|
| Room IRs | `Impulse_Responses/room_IRs/{0.2s,0.4s,0.6s,0.8s,real}` | 105 / 130 / 95 / 22 / 1 `.npy` |
| Stationary noise | `01_stationary_noise` | 289 DEMAND-style WAVs (2.6 GB), 40–300 s each |
| Non-stationary noise | `02_non-stationary_noise` | 1,948 WAVs |
| Device and occlusion IRs | `Impulse_Responses/device_IRs`, `Impulse_Responses/occlusion_IRs` | 17 and 30 |

## Loudness check (reproduce)

`results_2026-09-24/level_check/` holds the transcripts and `wer.py` stat files. The WAVs were
made like this:

```python
import numpy as np, soundfile as sf
x, _ = sf.read("mesolitica.wav"); y, _ = sf.read("rt0.4_snrnone_d0.wav")
g = y.std() / x.std()                                                        # -8.00 dB
sf.write("clean_quiet.wav", x * g, 16000, subtype="PCM_16")
sf.write("rt0.4_loud.wav", np.clip(y / g, -1, 1), 16000, subtype="PCM_16")   # clips peaks
```

Both were decoded with `transcribe_many.py --oracle-segments mesolitica.wav`.
