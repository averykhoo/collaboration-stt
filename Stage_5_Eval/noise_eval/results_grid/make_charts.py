"""Full-grid figures: the comparison table as an image, and WER against noise and against reverb.

Usage (from the repo root, collaboration-stt env):
    python Stage_5_Eval/noise_eval/results_grid/make_charts.py GT.txt

GT.txt is the reference (`gt_mesolitica.txt` from exported/Stage5.tar.gz). Writes to
results_grid/figures/:
  table.png         every cell of comparison.md, plus the two marginal blocks below
  wer_vs_snr.png    WER per SNR, averaged over the three RT60 buckets
  wer_vs_rt60.png   WER per RT60 bucket, averaged over the five SNR levels
  marginals.csv     the numbers behind both charts, with paired differences against ours

Marginals: each point is the mean of the cell WERs along the other axis (every cell weighted
equally; the clean run stands in for the RT60 none / SNR none cell). Every run has the same
1,046 reference words, so this is the same as pooling all runs with the clean run counted once
per draw. The CI comes from score.py's two-level bootstrap, extended across cells: one set of
utterance weights per replicate for every system and cell, and one draw resample per cell per
replicate shared by every system, so differences against ours are paired.

Check: the bootstrap consumes the RNG in score.py's order, so the per-cell CIs it computes must
equal the tracked result_*_cells.csv files exactly. The script asserts this before plotting.
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from wer_utt import load_allowed, read_hyp_words, read_ref_lines, utterance_counts  # noqa: E402

DAY1 = os.path.join(HERE, "..", "results_2026-09-24")
OUT = os.path.join(HERE, "figures")
B, SEED = 10000, 0
# (name, transcript dir, tracked cells CSV, colour, marker, on the charts). The table shows all six;
# the charts leave out noisy VAD, which is within ~2 pp of oracle VAD everywhere.
SYSTEMS = [
    ("Ours, oracle VAD", os.path.join(DAY1, "transcripts", "oracle_vad"), os.path.join(DAY1, "result_oracle_cells.csv"), "#2a78d6", "o", True),
    ("Ours, noisy VAD", os.path.join(DAY1, "transcripts", "noisy_vad"), os.path.join(DAY1, "result_vad_cells.csv"), None, None, False),
    ("Gemini Transcribe", os.path.join(HERE, "transcribe"), os.path.join(HERE, "result_transcribe_cells.csv"), "#eb6834", "s", True),
    ("Gemini Transcribe Live", os.path.join(HERE, "live"), os.path.join(HERE, "result_live_cells.csv"), "#1baf7a", "D", True),
    ("Gemma 4 E2B", os.path.join(HERE, "gemma_fp32"), os.path.join(HERE, "result_gemma_cells.csv"), "#eda100", "^", True),
    ("Whisper turbo + fallback", os.path.join(HERE, "whisper_fb"), os.path.join(HERE, "result_whisper_vs_oracle_cells.csv"), "#e87ba4", "v", True),
]
RTS, SNRS = (0.0, 0.4, 0.8), (float("inf"), 20.0, 10.0, 5.0, 0.0)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def ci(x):
    return np.quantile(x, 0.025, axis=0), np.quantile(x, 0.975, axis=0)


def bootstrap(gt):
    with open(os.path.join(DAY1, "manifest.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ref, allowed = read_ref_lines(gt), load_allowed()
    ids = ["clean"] + [r["id"] for r in rows]
    E, N = {}, {}
    for name, d, *_ in SYSTEMS:
        print("aligning", name, file=sys.stderr)
        C = np.stack([utterance_counts(ref, read_hyp_words(os.path.join(d, i + ".txt")), allowed) for i in ids])
        E[name], N[name] = C[..., 1] + C[..., 2] + C[..., 3], C[..., 0] + C[..., 1] + C[..., 3]

    # Same RNG consumption as score.py: utterance weights, then one draw resample per sorted cell.
    rng = np.random.default_rng(SEED)
    n = len(ref)
    Wu = rng.multinomial(n, np.full(n, 1 / n), size=B)
    cells = {}
    for k, r in enumerate(rows, 1):
        cells.setdefault((float(r["rt60"]), float(r["snr"])), []).append(k)
    Wd = {}
    for key, ks in sorted(cells.items()):
        Wd[key] = rng.multinomial(len(ks), np.full(len(ks), 1 / len(ks)), size=B)

    pt, bs = {}, {}  # [system][(rt, snr)] -> point WER, [B] replicates
    for name, *_ in SYSTEMS:
        Eb, Nb = Wu @ E[name].T, Wu @ N[name].T
        pt[name] = {(0.0, float("inf")): E[name][0].sum() / N[name][0].sum()}
        bs[name] = {(0.0, float("inf")): Eb[:, 0] / Nb[:, 0]}
        for key, ks in cells.items():
            ks = np.array(ks)
            pt[name][key] = E[name][ks].sum() / N[name][ks].sum()
            bs[name][key] = (Wd[key] * Eb[:, ks]).sum(1) / (Wd[key] * Nb[:, ks]).sum(1)
    return pt, bs


def check_against_tracked(pt, bs):
    for name, _, cells_csv, *_ in SYSTEMS:
        with open(cells_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (0.0, float("inf")) if r["rt60"] == "clean" else (float(r["rt60"]), float(r["snr"]))
                lo, hi = ci(bs[name][key])
                got = (pt[name][key], lo, hi)
                want = (float(r["wer"]), float(r["wer_lo"]), float(r["wer_hi"]))
                assert np.allclose(got, want, rtol=0, atol=1e-12), f"{name} {key}: {got} != tracked {want}"
    print("per-cell WER and CIs match all tracked cells CSVs", file=sys.stderr)


def marginals(pt, bs):
    """Rows of (axis, level, system, wer, lo, hi, vs, vs_lo, vs_hi)."""
    ours = SYSTEMS[0][0]
    out = []
    for axis, levels, group in (("snr", SNRS, lambda s: [(rt, s) for rt in RTS]),
                                ("rt60", RTS, lambda rt: [(rt, s) for s in SNRS])):
        for lv in levels:
            keys = group(lv)
            m_ours = np.mean([bs[ours][k] for k in keys], axis=0)
            for name, *_ in SYSTEMS:
                w = np.mean([pt[name][k] for k in keys])
                m = np.mean([bs[name][k] for k in keys], axis=0)
                d = m - m_ours
                vs = w - np.mean([pt[ours][k] for k in keys])
                out.append(dict(axis=axis, level=lv, system=name, wer=w, wer_lo=ci(m)[0], wer_hi=ci(m)[1],
                                vs=vs, vs_lo=ci(d)[0], vs_hi=ci(d)[1]))
    return out


def level_label(axis, lv):
    if axis == "snr":
        return "none" if lv == float("inf") else f"{lv:g}"
    return "none" if lv == 0.0 else f"{lv:g}s"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def chart(rows, axis, xlabel, title, subtitle, path):
    levels = SNRS if axis == "snr" else RTS
    shown = [s for s in SYSTEMS if s[5]]
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=160, facecolor=SURF)
    style(ax)
    x = np.arange(len(levels))
    dodge = np.linspace(-0.16, 0.16, len(shown))
    ends = []
    for (name, _, _, color, marker, _), dx in zip(shown, dodge):
        r = [next(o for o in rows if o["axis"] == axis and o["level"] == lv and o["system"] == name) for lv in levels]
        w = np.array([o["wer"] for o in r]) * 100
        lo = np.array([o["wer_lo"] for o in r]) * 100
        hi = np.array([o["wer_hi"] for o in r]) * 100
        ax.errorbar(x + dx, w, yerr=[w - lo, hi - w], color=color, linewidth=2, elinewidth=1.4, capsize=3,
                    marker=marker, markersize=8, markeredgecolor=SURF, markeredgewidth=2, label=name, zorder=3)
        # Hollow marker = significantly different from ours (paired CI excludes 0).
        for xi, o in zip(x + dx, r):
            if name != SYSTEMS[0][0] and (o["vs_hi"] < 0 or o["vs_lo"] > 0):
                ax.plot(xi, o["wer"] * 100, marker=marker, markersize=8, markerfacecolor=SURF,
                        markeredgecolor=color, markeredgewidth=2, zorder=4)
        ends.append([w[-1], name])
    # Direct labels at the right end, nudged apart so they do not collide.
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 3.2)
    for y, name in ends:
        ax.annotate(name, (x[-1] + 0.3, y), va="center", fontsize=9.5, color=INK, annotation_clip=False)
    ax.set_xticks(x, [level_label(axis, lv) for lv in levels])
    ax.set_xlim(-0.4, len(levels) - 0.65)
    ax.set_ylim(0, None)
    ax.set_xlabel(xlabel, color=INK2, fontsize=11)
    ax.set_ylabel("WER % (lower is better)", color=INK2, fontsize=11)
    fig.suptitle(title, x=0.06, ha="left", fontsize=14, color=INK, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=9.5, color=INK2, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(shown), fontsize=9, frameon=False,
              labelcolor=INK, handletextpad=0.3, columnspacing=1.2)
    fig.subplots_adjust(left=0.08, right=0.78, top=0.79, bottom=0.18)
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def table_image(rows, path):
    names = [s[0] for s in SYSTEMS]
    tracked = []
    for _, _, cells_csv, *_ in SYSTEMS:
        with open(cells_csv, newline="", encoding="utf-8") as f:
            tracked.append({(r["rt60"], r["snr"]): r for r in csv.DictReader(f)})
    grid = [("clean", "clean")] + [(rt, snr) for rt in ("0.0", "0.4", "0.8") for snr in ("inf", "20.0", "10.0", "5.0", "0.0")
                                   if (rt, snr) != ("0.0", "inf")]
    body = []  # (section or None, label cols, [(wer, lo, hi, marker)])
    for rt, snr in grid:
        rs = [t[(rt, snr)] for t in tracked]
        lab = ("clean", "clean") if rt == "clean" else (level_label("rt60", float(rt)), level_label("snr", float(snr)))
        vals = []
        for j, r in enumerate(rs):
            mk = "" if j == 0 else "▼" if float(r["vs_hi"]) < 0 else "▲" if float(r["vs_lo"]) > 0 else ""
            vals.append((float(r["wer"]), float(r["wer_lo"]), float(r["wer_hi"]), mk))
        body.append(("Every cell (RT60 bucket × SNR)", lab, vals))
    for axis, sec, levels in (("snr", "Mean over the 3 RT60 buckets", SNRS), ("rt60", "Mean over the 5 SNR levels", RTS)):
        for lv in levels:
            vals = []
            for j, name in enumerate(names):
                o = next(o for o in rows if o["axis"] == axis and o["level"] == lv and o["system"] == name)
                mk = "" if j == 0 else "▼" if o["vs_hi"] < 0 else "▲" if o["vs_lo"] > 0 else ""
                vals.append((o["wer"], o["wer_lo"], o["wer_hi"], mk))
            lab = ("all", level_label("snr", lv)) if axis == "snr" else (level_label("rt60", lv), "all")
            body.append((sec, lab, vals))

    cmap = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#9cc1ee", "#2a78d6", "#123e7a"])
    colw = [0.9, 0.9] + [2.05] * len(names)
    rowh, head, sech = 0.46, 0.95, 0.42
    nsec = len({b[0] for b in body})
    H = 1.15 + head + rowh * len(body) + sech * nsec + 0.9
    W = sum(colw) + 0.4
    fig = plt.figure(figsize=(W, H), dpi=160, facecolor=SURF)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    x0 = 0.2
    xs = np.cumsum([x0] + colw)
    ax.text(x0, 0.45, "Full-grid WER, all systems on all 71 files", fontsize=15, fontweight="bold", color=INK, va="center")
    ax.text(x0, 0.85, "WER % [95% CI], 10,000-replicate paired bootstrap over utterances and draws.   "
            "Bold = lowest in the row.   ▼ / ▲ = significantly better / worse than ours (oracle VAD); "
            "for noisy VAD, the VAD cost.", fontsize=9, color=INK2, va="center")
    y = 1.15
    for j, h in enumerate(["RT60", "SNR (dB)"] + names):
        ax.text((xs[j] + xs[j + 1]) / 2, y + head / 2, h.replace(" + ", "\n+ ").replace(", ", ",\n") if j >= 2 else h,
                ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    y += head
    ax.plot([x0, xs[-1]], [y, y], color=INK2, linewidth=1)
    cur = None
    for sec, lab, vals in body:
        if sec != cur:
            cur = sec
            ax.text(x0, y + sech * 0.62, sec, fontsize=9.5, fontstyle="italic", color=INK2, va="center")
            y += sech
        best = min(v[0] for v in vals)
        for j, t in enumerate(lab):
            ax.text((xs[j] + xs[j + 1]) / 2, y + rowh / 2, t, ha="center", va="center", fontsize=9.5, color=INK)
        for j, (w, lo, hi, mk) in enumerate(vals):
            c = cmap(min(w, 1.0))
            ax.add_patch(plt.Rectangle((xs[j + 2] + 0.02, y + 0.02), colw[j + 2] - 0.04, rowh - 0.04, color=c, linewidth=0))
            dark = min(w, 1.0) > 0.55
            txt = f"{100 * w:.1f}  [{100 * lo:.1f}, {100 * hi:.1f}]" + (f" {mk}" if mk else "")
            ax.text((xs[j + 2] + xs[j + 3]) / 2, y + rowh / 2, txt, ha="center", va="center", fontsize=9,
                    color="#ffffff" if dark else INK, fontweight="bold" if abs(w - best) < 1e-12 else "normal")
        y += rowh
    ax.text(x0, y + 0.35, "Cell colour = WER (lighter is better). Each mean row weights its cells equally; the clean run is the "
            "none/none cell. Charts leave out noisy VAD, which is within ~2 pp of oracle VAD.", fontsize=8.5, color=INK2, va="center")
    ax.text(x0, y + 0.62, "Source: result_*_cells.csv (every cell) and figures/marginals.csv (means), "
            "built by results_grid/make_charts.py.", fontsize=8.5, color=INK2, va="center")
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    os.makedirs(OUT, exist_ok=True)
    pt, bs = bootstrap(sys.argv[1])
    check_against_tracked(pt, bs)
    rows = marginals(pt, bs)
    with open(os.path.join(OUT, "marginals.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sub = ("Mean of the cell WERs across the {}, every cell weighted equally.\n"
           "95% CI from the paired bootstrap over utterances and draws. Hollow marker = significantly different\n"
           "from ours (the paired CI of the difference excludes 0).")
    chart(rows, "snr", "SNR (dB); none = no added noise", "WER against noise, averaged over reverb",
          sub.format("three RT60 buckets (none, 0.4s, 0.8s)"), os.path.join(OUT, "wer_vs_snr.png"))
    chart(rows, "rt60", "RT60 bucket (reverb time); none = no reverb", "WER against reverb, averaged over noise",
          sub.format("five SNR levels (none, 20, 10, 5, 0 dB)"), os.path.join(OUT, "wer_vs_rt60.png"))
    table_image(rows, os.path.join(OUT, "table.png"))
    print("wrote", OUT, file=sys.stderr)


if __name__ == "__main__":
    main()
