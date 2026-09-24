"""How far the VAD's segments on noisy audio drift from the clean-audio (oracle) segments.

Usage (any env with numpy):
    python noise_eval/vad_stats.py MANIFEST.csv VAD_PRED_DIR ORACLE_SEGMENTS.json OUT.md

Per run, against the oracle segments:
  * coverage: share of oracle speech time the noisy segments also cover (missed speech = 1 - coverage);
  * extra: noisy segment time outside the oracle segments, in seconds;
  * segs: number of segments decoded.
Reported per (rt60, snr) cell as mean ± sd over draws.
"""
import csv
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def mask(segs, n):
    """Boolean timeline at 10 ms resolution."""
    m = np.zeros(n, dtype=bool)
    for b, e in segs:
        m[max(0, int(round(b * 100))):int(round(e * 100))] = True
    return m


def main(manifest, pred_dir, oracle_json, out_md):
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(oracle_json) as f:
        oracle = json.load(f)["segments"]
    n = int(max(e for _, e in oracle) * 100) + 3000  # oracle end plus 30 s headroom
    om = mask(oracle, n)

    cells = {}
    for r in rows:
        with open(os.path.join(pred_dir, r["id"] + ".segments.json")) as f:
            segs = json.load(f)
        m = mask(segs, n)
        cov = (m & om).sum() / om.sum()
        extra = (m & ~om).sum() / 100
        cells.setdefault((float(r["rt60"]), float(r["snr"])), []).append((cov, extra, len(segs)))

    sd = lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0
    lines = [f"Oracle (clean-audio VAD): {len(oracle)} segments, {om.sum() / 100:.1f} s of speech.", "",
             "| RT60 (s) | SNR (dB) | draws | coverage of oracle speech % | extra time s | segments |",
             "|---|---|---|---|---|---|"]
    for (rt60, snr), v in sorted(cells.items()):
        c, x, s = (np.array(t, dtype=float) for t in zip(*v))
        lines.append(f"| {rt60} | {snr} | {len(v)} | {100 * c.mean():.1f} ± {100 * sd(c):.1f} "
                     f"| {x.mean():.1f} ± {sd(x):.1f} | {s.mean():.1f} ± {sd(s):.1f} |")
    md = "\n".join(lines)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main(*sys.argv[1:5])
