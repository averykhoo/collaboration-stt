"""Per-utterance attribution of wer.py's corpus-level WER.

wer.py aligns the whole hypothesis against the whole reference as one word
sequence, because predict.py's VAD segments do not line up with the reference
lines. This module reuses that exact alignment and span rules, but charges each
error to the reference line (utterance) it belongs to, so the corpus WER can be
bootstrapped over utterances. Needs the wer.py column-0 traceback fix, without
which leading reference words drop out of the alignment.

Invariant (asserted in `utterance_counts`): the per-utterance sums reproduce
wer.py's M/S/I/D totals exactly.
"""
import os
import sys
from typing import List, Tuple

import numpy as np

STAGE5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if STAGE5 not in sys.path:
    sys.path.insert(0, STAGE5)

from wer import AccuracyStatistics  # noqa: E402
from k2_scripts.gen_utils import my_reader, continuous_replace  # noqa: E402


def _clean(line: str) -> List[str]:
    line = line.strip().replace("<unk>", "")
    return continuous_replace("  ", " ", line).split()


def read_ref_lines(path: str) -> List[List[str]]:
    """Reference as a list of non-empty utterances, tokenised as wer.py does."""
    with my_reader(path) as r:
        return [w for w in (_clean(l) for l in r) if w]


def read_hyp_words(path: str) -> List[str]:
    with my_reader(path) as r:
        return [x for l in r for x in _clean(l)]


def load_allowed(stage5: str = STAGE5):
    """The three allowed_* lists, parsed as wer.py's __main__ does."""
    def lines(name):
        with my_reader(os.path.join(stage5, name)) as r:
            return [l.strip() for l in r if l.strip()]
    reps = [tuple(l.split(",", 1)) for l in lines("allowed_replacements.txt")]
    return reps, lines("allowed_insertions.txt"), lines("allowed_deletions.txt")


def utterance_counts(ref_lines: List[List[str]], hyp: List[str], allowed) -> np.ndarray:
    """Return an int array [n_utt, 4] of (M, S, I, D) per reference utterance.

    Mirrors AccuracyStatistics.accumulate span-for-span. Each counted error goes
    to the utterance of its own reference word; an insertion goes to the
    utterance of the last reference word seen before it (utterance 0 if none).
    """
    reps, ins, dels = allowed
    ref = [w for u in ref_lines for w in u]
    utt_of_ref = [i for i, u in enumerate(ref_lines) for _ in u]

    stats = AccuracyStatistics(reps, ins, dels)
    pairs = stats._AccuracyStatistics__global_alignment(ref, hyp)

    # Utterance tag per aligned pair.
    tags, k, last = [], 0, 0
    for r_w, _ in pairs:
        if r_w is not None:
            last = utt_of_ref[k]
            k += 1
        tags.append(last)
    assert k == len(ref), "alignment dropped reference words (wer.py column-0 traceback bug)"

    out = np.zeros((len(ref_lines), 4), dtype=np.int64)
    n, start = len(pairs), 0
    while start < n:
        nxt = start
        while nxt < n:
            r_w, t_w = pairs[nxt]
            if r_w is not None and t_w is not None and r_w == t_w:
                break
            nxt += 1
        if nxt == start:
            out[tags[start], 0] += 1
            start += 1
            continue
        span = pairs[start:nxt]
        ref_idx = [i for i, (r_w, _) in enumerate(span, start) if r_w is not None]
        u = tags[ref_idx[0]] if ref_idx else tags[start]
        mref = [r_w for r_w, _ in span if r_w is not None]
        mtr = [t_w for _, t_w in span if t_w is not None]
        if not mref:
            counted = " ".join(mtr) not in stats._allowed_insertions
        elif not mtr:
            counted = " ".join(mref) not in stats._allowed_deletions
        else:
            counted = (" ".join(mref), " ".join(mtr)) not in stats._allowed_replacements
        if counted:
            # Charge each error to its own utterance, so a span that runs across many
            # utterances (e.g. a near-empty hypothesis) is not dumped on the first one.
            for i, (r_w, t_w) in enumerate(span, start):
                out[tags[i], 2 if r_w is None else 3 if t_w is None else 1] += 1
        elif mref and mtr:
            out[u, 0] += 1  # an allowed substitution counts as one match
        start = nxt

    # Must reproduce wer.py exactly.
    ref_stats = AccuracyStatistics(reps, ins, dels)
    ref_stats.accumulate(ref, hyp)
    tot = tuple(int(x) for x in out.sum(0))
    want = (ref_stats._num_matches, ref_stats._num_substitutions,
            ref_stats._num_insertions, ref_stats._num_deletions)
    assert tot == want, f"attribution {tot} != wer.py {want}"
    return out


def wer_of(counts: np.ndarray) -> float:
    """Corpus WER from [n_utt, 4] (M,S,I,D) counts, as wer.py computes it."""
    m, s, i, d = counts.sum(0)
    return (s + i + d) / (m + s + d)
