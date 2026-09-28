"""Build the consolidated full-grid comparison table from the tracked score.py cells CSVs.

Usage (from the repo root):
    python Stage_5_Eval/noise_eval/results_grid/make_comparison.py > Stage_5_Eval/noise_eval/results_grid/comparison.md

Each cell is WER % [95% CI]. The paired difference against our oracle-VAD run decides the marker:
  ▼ significantly better than ours (the whole 95% CI of the difference is below 0),
  ▲ significantly worse than ours (the whole CI is above 0).
For "ours, noisy VAD" the marker is the VAD cost against our oracle-VAD run.
Bold marks the lowest WER in each row.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DAY1 = os.path.join(HERE, "..", "results_2026-09-24")
SYSTEMS = [  # (column, cells CSV, CSV column holding the paired difference vs ours-oracle)
    ("Ours, oracle VAD", os.path.join(DAY1, "result_oracle_cells.csv"), None),
    ("Ours, noisy VAD", os.path.join(DAY1, "result_vad_cells.csv"), "vs"),
    ("Gemini Transcribe", os.path.join(HERE, "result_transcribe_cells.csv"), "vs"),
    ("Gemini Transcribe Live", os.path.join(HERE, "result_live_cells.csv"), "vs"),
    ("Gemma 4 E2B", os.path.join(HERE, "result_gemma_cells.csv"), "vs"),
    ("Whisper turbo + fallback", os.path.join(HERE, "result_whisper_vs_oracle_cells.csv"), "vs"),
]
ORDER = [("clean", "clean")] + [(rt, snr) for rt in ("0.0", "0.4", "0.8") for snr in ("inf", "20.0", "10.0", "5.0", "0.0")
                                if (rt, snr) != ("0.0", "inf")]


def label(rt, snr):
    if rt == "clean":
        return "clean", "clean"
    return ("none" if rt == "0.0" else rt + "s"), ("none" if snr == "inf" else str(int(float(snr))))


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["rt60"], r["snr"]): r for r in csv.DictReader(f)}


tables = [(name, load(path), vs) for name, path, vs in SYSTEMS]
print("| RT60 bucket | SNR (dB) | " + " | ".join(name for name, _, _ in tables) + " |")
print("|---|---|" + "---|" * len(tables))
for key in ORDER:
    rows = [t.get(key) for _, t, _ in tables]
    best = min(float(r["wer"]) for r in rows if r)
    cells = []
    for (name, _, vs), r in zip(tables, rows):
        if r is None:
            cells.append("—")
            continue
        w = float(r["wer"])
        txt = f"{100 * w:.1f} [{100 * float(r['wer_lo']):.1f}, {100 * float(r['wer_hi']):.1f}]"
        if abs(w - best) < 1e-12:
            txt = f"**{txt}**"
        if vs:
            lo, hi = float(r[vs + "_lo"]), float(r[vs + "_hi"])
            txt += " ▼" if hi < 0 else " ▲" if lo > 0 else ""
        cells.append(txt)
    print(f"| {' | '.join(label(*key))} | " + " | ".join(cells) + " |")
