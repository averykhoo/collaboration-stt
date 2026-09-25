# Runbook: Stage 5 speech-to-text on a local CPU

This runbook covers transcribing an audio file with the Stage 5 Malay ASR model on a
Windows laptop with no GPU. The commands are for **Git Bash**. It was verified end to
end on 2026-09-24 on an i7-1365U laptop with 32 GB RAM.

## ⚠ Before you start

- **Do not run `sh/main.sh` as-is.** It deletes `asr_model/` and `input/`, then copies
  from `/mount/data/alx/...`, which is the original author's Linux machine.
- **Keep torch and torchaudio below 2.9.** From 2.9, `torchaudio.load` goes through
  TorchCodec, which needs FFmpeg DLLs on Windows. The Stage 3 `requirements.txt` pins
  2.9.0, so don't reuse it here.
- **Run `predict.py` from inside `Stage_5_Eval/`.** The VAD checkpoint path and the
  Python imports are relative to that folder.

## 1. Create the Python environment

This needs a one-off download of about 1 GB. The environment is named after the repo.

```bash
CONDA=C:/Users/user/anaconda3/Scripts/conda.exe
$CONDA env list | grep -q '^collaboration-stt ' || $CONDA create -y -n collaboration-stt python=3.12

PY=C:/Users/user/anaconda3/envs/collaboration-stt/python.exe
$PY -m pip install "torch<2.9" "torchaudio<2.9" numpy scipy soundfile pytorch_lightning \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

Check it. This should print `2.8.0+cpu False`:

```bash
$PY -c "import torch, torchaudio, pytorch_lightning; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Put the model files in place

The weights are **not in git**. They come from the `exported/asr_model.tar.gz`
handover tarball, which is 243 MB. `exported/Stage5.tar.gz` holds an identical copy.
From the repo root:

```bash
tar xzf exported/asr_model.tar.gz -C Stage_5_Eval
ls Stage_5_Eval/asr_model    # bpe.model  pretrained.pt  tokens.txt
```

`predict.py` reads `asr_model/pretrained.pt` and `asr_model/tokens.txt` by default.
Don't commit this folder, because `pretrained.pt` is 263 MB.

## 3. Prepare the input

- **Audio format.** Use WAV. Other sample rates are resampled to 16 kHz automatically.
  Only 16 kHz mono has been tested.
- **Path list.** `--wav_pathes` takes a text file with **one absolute path per line**,
  not the audio file itself. On Windows, write paths as `C:/...`.

```bash
cd Stage_5_Eval
echo "$(cd /path/to/audio && pwd -W)/my_clip.wav" > my_paths.txt
cat my_paths.txt    # e.g. C:/Users/user/Music/my_clip.wav
```

## 4. Transcribe

```bash
$PY predict.py --wav_pathes my_paths.txt --out_predict my_predict.txt
```

The script cuts the audio into speech segments of up to 30 seconds and writes **one
line of text per segment**. If the path list has several files, all their lines go into
the same output with no separator. Run one file at a time if you need to tell them apart.

The script prints two warnings, and both are harmless. One is about Lightning upgrading
the VAD checkpoint in memory. The other is about torchaudio's 2.9 change.

Speed on 2026-09-24: a 593-second clip took 103 seconds, including model load.

## 5. Score it (optional)

For WER against a reference transcript:

```bash
$PY wer.py reference.txt my_predict.txt \
  allowed_replacements.txt allowed_insertions.txt allowed_deletions.txt my_stat.txt
head -2 my_stat.txt
```

## 6. Verify the setup against the reference run

Use this to confirm that a new setup behaves exactly like the original GPU run. The
reference clip and outputs are in `exported/Stage5.tar.gz`. From the repo root:

```bash
mkdir -p .scratch/verify
tar xzf exported/Stage5.tar.gz -C .scratch/verify \
  Stage_5_Eval/input/mesolitica.wav Stage_5_Eval/input/gt_mesolitica.txt \
  Stage_5_Eval/inter/mesolitica/predict/predict_mesolitica.txt
V=$(cd .scratch/verify/Stage_5_Eval && pwd -W)

cd Stage_5_Eval
echo "$V/input/mesolitica.wav" > ../.scratch/verify/paths.txt
$PY predict.py --wav_pathes ../.scratch/verify/paths.txt --out_predict ../.scratch/verify/predict.txt
cmp ../.scratch/verify/predict.txt "$V/inter/mesolitica/predict/predict_mesolitica.txt" && echo MATCH
```

`MATCH` means the setup is good. On 2026-09-24 the output was byte-identical and WER was
35.5%. Any difference usually means the model weights didn't load fully.
`predict.py` loads them with `strict=False`, so a mismatch fails silently instead of
raising an error.

## Cleanup

- Delete `.scratch/verify/` and your own output files when you're done.
- Running Python creates `__pycache__/` folders under `Stage_5_Eval/`. They are safe to
  delete and shouldn't be committed.

## Noise and reverb evaluation

`noise_eval/` runs this model on far-field versions of the Mesolitica clip, built with the
Farfield Audio Synthesis Toolkit, and scores WER with bootstrap confidence intervals. It covers
two conditions: the normal noisy VAD, and an oracle that reuses the clean-audio VAD segments.
Results, method and reproduction steps are in `noise_eval/RESULTS.md`.

## Baseline models: Whisper, Gemini, Gemma

These runners transcribe the same noise-eval files with other models, for comparison. Each
script's docstring has the full detail. Results from 2026-09-25 are in `noise_eval/RESULTS.md`
and `noise_eval/results_2026-09-25/`. Run everything from the **repo root**.

| Runner | Model | Where it runs | Time per file (2026-09-25) |
|---|---|---|---|
| `whisper_transcribe.py` | `mesolitica/Malaysian-whisper-large-v3-turbo-v3` | laptop CPU | 22–41 min with `--fallback` |
| `gemma_transcribe.py` | `google/gemma-4-E2B-it` (or E4B / 12B) | laptop CPU | 89–106 min in bf16 |
| `gemini_transcribe.py` | `gemini-3.5-transcribe`, or a Flash model with `--prompt malay` | Gemini API | 1–5 min |
| `gemini_live_transcribe.py` | `gemini-3.5-transcribe-live` | Gemini Live API (WebSocket) | about 3 min at `--speed 4` |

### 1. Extra packages

The runners need more than the base environment from section 1. The versions below were
installed on 2026-09-25. **Check with `pip install --dry-run` first.** Nothing here may change
torch, torchaudio, numpy or transformers.

```bash
$PY -m pip install transformers pillow          # transformers 5.17.0, pillow 12.3.0
$PY -m pip install "torchvision==0.23.0" \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

- **Gemma needs torchvision.** Its processor imports it. A plain `pip install torchvision`
  would pull a newer torch, so pin 0.23.0 (it matches torch 2.8.0) from the CPU index.
- **aiohttp** is needed by the Live runner; it was already in the environment (3.14.3).
- **Weights download on first use** into the Hugging Face cache: Whisper about 1.6 GB, Gemma E2B
  10.2 GB. Neither checkpoint is gated.
- **Skip faster-whisper.** The int8 Whisper path (`--backend ct2`) isn't a faithful stand-in:
  it matched fp32 on only 13 of 42 segments. It's kept for the record only.

### 2. Inputs

The runners take the noise-eval manifest **plus a `clean` row**, and the oracle segments:

```bash
E=.scratch/noise_eval
M=.scratch/manifest_all.csv
S=Stage_5_Eval/noise_eval/results_2026-09-24/oracle.segments.json
{ cat $E/wav/manifest.csv; echo "clean,$(cd .scratch/meso/Stage_5_Eval/input && pwd -W)/mesolitica.wav,,,,,,,"; } > $M
```

- **Where the audio comes from.** `$E/wav/` is the output of `augment.py`, and
  `.scratch/meso/` is where `exported/Stage5.tar.gz` gets unpacked. `noise_eval/RESULTS.md` →
  Reproduce covers both.
- **Choosing files.** Use `--ids id1,id2,...` to run a subset.

### 3. Gemini API keys

- **Where keys live:** `~/.gemini_api_key` by default. Both Gemini runners take
  `--key-file PATH` for a second key.
- **Never put a key in the repo or in a command's output.** Keys pasted into chat must be
  rotated.
- **Free-tier daily limits are per key and per model:** 25 requests for Transcribe and 20 for
  each Flash model. Transcribe Live has no daily limit.
- **Failed requests use up quota too.** On 2026-09-25, requests that failed with HTTP 503 still
  counted towards the daily limit.

### 4. Run

Every runner can be resumed: rerun the same command and it skips finished files. The Gemma
runner saves after every segment.

```bash
# Whisper: always use --fallback. Pure greedy decoding loops on noisy audio.
$PY Stage_5_Eval/noise_eval/whisper_transcribe.py $M $S $OUT --fallback --ids ...

# Gemma 4 (bf16, about 12 GB RAM)
HF_HUB_DISABLE_SYMLINKS_WARNING=1 $PY Stage_5_Eval/noise_eval/gemma_transcribe.py $M $S $OUT --ids ...

# Gemini Transcribe (no prompt). Default spacing is 65 s for its 10K tokens/min limit.
$PY Stage_5_Eval/noise_eval/gemini_transcribe.py $M $S $OUT --ids ...

# Gemini Flash models: prompted, temperature 0. --thinking low is optional and did not help.
$PY Stage_5_Eval/noise_eval/gemini_transcribe.py $M $S $OUT --model gemini-3.6-flash \
  --prompt malay --min-interval 15 --ids ...

# Gemini Transcribe Live: pace the stream. Unpaced streaming hits the per-minute token limit.
$PY Stage_5_Eval/noise_eval/gemini_live_transcribe.py $M $S $OUT --speed 4 --ids ...
```

- **HTTP 503 "high demand"** is common on the Flash models. `gemini_transcribe.py` retries
  after 1, 2 and 4 minutes, then exits. Just rerun it later.
- **Exit code 3** means the daily quota is used up.

### 5. Score

```bash
GT=.scratch/meso/Stage_5_Eval/input/gt_mesolitica.txt
ORACLE=Stage_5_Eval/noise_eval/results_2026-09-24/transcripts/oracle_vad   # our model
$PY Stage_5_Eval/noise_eval/score.py MANIFEST $OUT $GT $OUT/result --vs $ORACLE
$PY Stage_5_Eval/noise_eval/score_files.py $OUT [$OUT2 ...] > scores.csv   # per file, no CIs
```

- **`score.py` needs the full set.** It expects a transcript for every manifest row plus
  `clean.txt`. The manifest must not contain the `clean` row itself.
- **`score_files.py` handles partial and retry runs.** It also flags outputs over 5,000 words as
  "runaway" instead of aligning them.
- **Don't feed looping output to `score.py`.** Aligning a 65K-word output took over 16 GB of
  RAM and pushed a running Gemma job out of memory. Check word counts first
  (`wc -w $OUT/*.txt`; the reference is 1,046 words).

### 6. Sharing the laptop

- **Run one CPU model at a time unless you've checked RAM.** Whisper fp32 needs about 4.5 GB,
  Gemma E2B bf16 about 12 GB, and each uses 4 threads. Two jobs at once slow each other and
  skew both timings.
- **The Gemini runners are network-bound** and can run alongside anything.
- **Copy results into a tracked folder the same day.** Record each finished run in a dated
  `noise_eval/results_<date>/` folder and in `RESULTS.md`. `.scratch/` is not in git.
