"""Score clean vs augmented transcripts with bootstrap confidence intervals.

Usage (collaboration-stt env):
    python noise_eval/score.py MANIFEST.csv PRED_DIR GT.txt OUT_PREFIX [--boot 10000] [--seed 0]
                               [--vs OTHER_PRED_DIR]

MANIFEST.csv columns: id, rt60, snr, draw. The clean run has id `clean` and is
not listed in the manifest. PRED_DIR/<id>.txt are transcripts from
transcribe_many.py.

WER is corpus-level, computed as wer.py does it: (S+I+D)/(M+S+D) over the whole
clip. Uncertainty comes from a two-level paired bootstrap:
  * resample the reference utterances (the same resample is used for every
    condition in a replicate, so clean-vs-noisy differences are paired);
  * within each (rt60, snr) cell, also resample the augmentation draws
    (different IR, noise clip and seed per draw).
The CI therefore covers both test-set sampling and augmentation randomness. The
clean run is deterministic, so its CI comes from utterance resampling alone.
Point estimates pool the counts over all draws and utterances.

--vs OTHER_PRED_DIR adds a paired column: this condition's WER minus the other
directory's, per cell, under the same utterance and draw resamples (e.g. noisy
VAD vs oracle VAD, where the difference is the cost of VAD errors).
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")  # the table has Δ and ±; Windows consoles default to cp1252

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wer_utt import load_allowed, read_hyp_words, read_ref_lines, utterance_counts  # noqa: E402


def ci(x, level=0.95):
    a = (1 - level) / 2
    return np.quantile(x, a), np.quantile(x, 1 - a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("pred_dir")
    p.add_argument("gt")
    p.add_argument("out_prefix")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vs")
    a = p.parse_args()

    with open(a.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ref = read_ref_lines(a.gt)
    allowed = load_allowed()

    def counts(pred_dir, run_id):
        return utterance_counts(ref, read_hyp_words(os.path.join(pred_dir, run_id + ".txt")), allowed)

    ids = ["clean"] + [r["id"] for r in rows]
    C = np.stack([counts(a.pred_dir, i) for i in ids])  # [run, utt, 4]
    E = C[..., 1] + C[..., 2] + C[..., 3]                # [run, utt]
    N = C[..., 0] + C[..., 1] + C[..., 3]
    if a.vs:
        C2 = np.stack([counts(a.vs, i) for i in ids])
        E2 = C2[..., 1] + C2[..., 2] + C2[..., 3]
        N2 = C2[..., 0] + C2[..., 1] + C2[..., 3]

    rng = np.random.default_rng(a.seed)
    n_utt = len(ref)
    Wu = rng.multinomial(n_utt, np.full(n_utt, 1 / n_utt), size=a.boot)  # [B, utt]
    Eb, Nb = Wu @ E.T, Wu @ N.T                                           # [B, run]
    if a.vs:
        Eb2, Nb2 = Wu @ E2.T, Wu @ N2.T

    clean_b = Eb[:, 0] / Nb[:, 0]
    clean_pt = E[0].sum() / N[0].sum()

    cells = {}
    for k, r in enumerate(rows, 1):
        cells.setdefault((float(r["rt60"]), float(r["snr"])), []).append(k)

    out = [dict(rt60="clean", snr="clean", draws=1, wer=clean_pt,
                wer_lo=ci(clean_b)[0], wer_hi=ci(clean_b)[1],
                draw_mean=clean_pt, draw_sd=0.0,
                delta=0.0, delta_lo=0.0, delta_hi=0.0)]
    if a.vs:
        vs_b = clean_b - Eb2[:, 0] / Nb2[:, 0]
        out[0].update(vs=clean_pt - E2[0].sum() / N2[0].sum(), vs_lo=ci(vs_b)[0], vs_hi=ci(vs_b)[1])
    for (rt60, snr), ks in sorted(cells.items()):
        ks = np.array(ks)
        K = len(ks)
        Wd = rng.multinomial(K, np.full(K, 1 / K), size=a.boot)          # [B, draw]
        cell_b = (Wd * Eb[:, ks]).sum(1) / (Wd * Nb[:, ks]).sum(1)
        d_b = cell_b - clean_b
        per_draw = E[ks].sum(1) / N[ks].sum(1)
        pt = E[ks].sum() / N[ks].sum()
        out.append(dict(rt60=rt60, snr=snr, draws=K, wer=pt,
                        wer_lo=ci(cell_b)[0], wer_hi=ci(cell_b)[1],
                        draw_mean=per_draw.mean(), draw_sd=per_draw.std(ddof=1) if K > 1 else 0.0,
                        delta=pt - clean_pt, delta_lo=ci(d_b)[0], delta_hi=ci(d_b)[1]))
        if a.vs:
            vs_b = cell_b - (Wd * Eb2[:, ks]).sum(1) / (Wd * Nb2[:, ks]).sum(1)
            out[-1].update(vs=pt - E2[ks].sum() / N2[ks].sum(), vs_lo=ci(vs_b)[0], vs_hi=ci(vs_b)[1])

    # Per-run table, for inspection.
    with open(a.out_prefix + "_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "rt60", "snr", "draw", "M", "S", "I", "D", "wer"])
        meta = [dict(rt60="", snr="", draw="")] + rows
        for i, m, c in zip(ids, meta, C):
            M, S, I, D = c.sum(0)
            w.writerow([i, m["rt60"], m["snr"], m["draw"], M, S, I, D, f"{(S + I + D) / (M + S + D):.4f}"])

    with open(a.out_prefix + "_cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    pct = lambda x: f"{100 * x:.1f}"
    spct = lambda x: f"{100 * x:+.1f}"
    vs_name = os.path.basename(os.path.normpath(a.vs)) if a.vs else ""
    lines = ["| RT60 (s) | SNR (dB) | draws | WER % [95% CI] | per-draw mean ± sd | ΔWER vs clean, pp [95% CI] |"
             + (f" ΔWER vs {vs_name}, pp [95% CI] |" if a.vs else ""),
             "|---|---|---|---|---|---|" + ("---|" if a.vs else "")]
    for o in out:
        lines.append(f"| {o['rt60']} | {o['snr']} | {o['draws']} | {pct(o['wer'])} [{pct(o['wer_lo'])}, {pct(o['wer_hi'])}] "
                     f"| {pct(o['draw_mean'])} ± {pct(o['draw_sd'])} "
                     + ("| — |" if o["rt60"] == "clean" else
                     f"| {spct(o['delta'])} [{spct(o['delta_lo'])}, {spct(o['delta_hi'])}] |")
                     + (f" {spct(o['vs'])} [{spct(o['vs_lo'])}, {spct(o['vs_hi'])}] |" if a.vs else ""))
    md = "\n".join(lines)
    with open(a.out_prefix + "_table.md", "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print(f"\nbootstrap: {a.boot} replicates, {n_utt} utterances, seed {a.seed}")


if __name__ == "__main__":
    main()
