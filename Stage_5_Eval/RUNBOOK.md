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
