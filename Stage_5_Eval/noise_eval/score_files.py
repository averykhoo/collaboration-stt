"""Per-file corpus WER (no bootstrap) for every transcript dir given, as CSV on stdout.

Usage (collaboration-stt env, from the repo root):
    python Stage_5_Eval/noise_eval/score_files.py DIR [DIR ...] > scores.csv

For partial or retry runs that score.py cannot take (it needs every manifest row).
Outputs longer than 5,000 words are reported as "runaway" and not aligned: aligning
a 65K-word looping output took over 16 GB of RAM on 2026-09-25. The finish column is
the Gemini finishReason from raw/<id>.json, or the block reason when there is none.
The reference is gt_mesolitica.txt from exported/Stage5.tar.gz.
"""
import csv, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wer_utt import load_allowed, read_hyp_words, read_ref_lines, utterance_counts, wer_of
ref = read_ref_lines(".scratch/meso/Stage_5_Eval/input/gt_mesolitica.txt")
allowed = load_allowed()
w = csv.writer(sys.stdout, lineterminator="\n")
w.writerow(["dir", "id", "hyp_words", "M", "S", "I", "D", "wer", "finish"])
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.txt"))):
        i = os.path.basename(f)[:-4]
        hyp = read_hyp_words(f)
        fin = ""
        rp = os.path.join(d, "raw", i + ".json")
        if os.path.exists(rp):
            resp = json.load(open(rp, encoding="utf-8")).get("responses", [{}])
            if resp:
                c = resp[0].get("candidates", [{}])[0]
                fin = c.get("finishReason") or ("blocked:" + str(resp[0].get("promptFeedback", {}).get("blockReason")))
        if len(hyp) > 5000:
            w.writerow([d, i, len(hyp), "", "", "", "", "runaway", fin]); continue
        u = utterance_counts(ref, hyp, allowed)
        w.writerow([d, i, len(hyp), *u.sum(0).tolist(), round(100 * wer_of(u), 1), fin])
    sys.stdout.flush()
