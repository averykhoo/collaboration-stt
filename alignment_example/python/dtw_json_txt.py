#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Align a ground-truth TXT transcript with a segmented STT JSON using
dynamic programming (via jiwer), and produce per-segment ground-truth text
plus per-segment CER (character error rate).

CER is printed as a percentage string, e.g.: "3.85%".
"""

import sys
import json
from typing import List, Tuple

try:
    import jiwer
except ImportError:
    print("ERROR: This script requires the 'jiwer' package. Install with:", file=sys.stderr)
    print("       pip install jiwer", file=sys.stderr)
    sys.exit(1)


def compute_cer(hyp: str, ref: str) -> float:
    """
    Compute CER between hypothesis and reference strings.
    Returns a float in [0.0–1.0].
    """
    if not ref.strip():
        return 0.0 if not hyp.strip() else 1.0
    return jiwer.cer(ref, hyp)


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_json_segments(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["segments"]


def tokenize_simple(text: str) -> List[str]:
    return [w for w in text.split() if w]


def build_global_hypothesis_and_boundaries(segments: List[dict]) -> Tuple[List[str], List[Tuple[int, int]]]:
    global_words: List[str] = []
    seg_ranges: List[Tuple[int, int]] = []

    for seg in segments:
        sent = seg.get("sentence", "") or ""
        words = tokenize_simple(sent)
        start_idx = len(global_words)
        global_words.extend(words)
        end_idx = len(global_words)
        seg_ranges.append((start_idx, end_idx))

    return global_words, seg_ranges


def compute_alignment(ref_words: List[str], hyp_words: List[str]):
    ref_str = " ".join(ref_words)
    hyp_str = " ".join(hyp_words)
    out = jiwer.process_words(ref_str, hyp_str)
    return out.alignments[0]


def build_hyp_to_ref_map(num_hyp: int, alignments) -> Tuple[List[int], List[int]]:
    hyp_to_ref_start = [None] * num_hyp
    hyp_to_ref_end = [None] * num_hyp

    for ch in alignments:
        ch_type = getattr(ch, "type", None)
        rs, re = ch.ref_start_idx, ch.ref_end_idx
        hs, he = ch.hyp_start_idx, ch.hyp_end_idx

        if ch_type in ("equal", "replace", "substitute"):
            n_ref = re - rs
            n_hyp = he - hs

            if n_ref == n_hyp and n_hyp > 0:
                for k in range(n_hyp):
                    h_idx = hs + k
                    r_idx = rs + k
                    if hyp_to_ref_start[h_idx] is None:
                        hyp_to_ref_start[h_idx] = r_idx
                        hyp_to_ref_end[h_idx] = r_idx + 1
                    else:
                        hyp_to_ref_start[h_idx] = min(hyp_to_ref_start[h_idx], r_idx)
                        hyp_to_ref_end[h_idx] = max(hyp_to_ref_end[h_idx], r_idx + 1)
            else:
                for h_idx in range(hs, he):
                    if hyp_to_ref_start[h_idx] is None:
                        hyp_to_ref_start[h_idx] = rs
                        hyp_to_ref_end[h_idx] = re
                    else:
                        hyp_to_ref_start[h_idx] = min(hyp_to_ref_start[h_idx], rs)
                        hyp_to_ref_end[h_idx] = max(hyp_to_ref_end[h_idx], re)

        elif ch_type in ("insert", "delete"):
            continue

    return hyp_to_ref_start, hyp_to_ref_end


def extract_segment_ground_truth(
    seg_ranges: List[Tuple[int, int]],
    ref_words: List[str],
    hyp_to_ref_start: List[int],
    hyp_to_ref_end: List[int],
) -> List[str]:

    results: List[str] = []

    for (hs, he) in seg_ranges:
        rs_candidates = []
        re_candidates = []

        for h_idx in range(hs, he):
            rs = hyp_to_ref_start[h_idx]
            re = hyp_to_ref_end[h_idx]
            if rs is not None and re is not None:
                rs_candidates.append(rs)
                re_candidates.append(re)

        if not rs_candidates:
            results.append("")
            continue

        rs_seg = min(rs_candidates)
        re_seg = max(re_candidates)

        gt_tokens = ref_words[rs_seg:re_seg]
        results.append(" ".join(gt_tokens))

    return results


def main():
    if len(sys.argv) != 4:
        print("Usage: python dtw_txt_json.py in_txt in_json out_json", file=sys.stderr)
        sys.exit(1)

    in_txt = sys.argv[1]
    in_json = sys.argv[2]
    out_json = sys.argv[3]

    ref_text = read_txt(in_txt)
    segments = read_json_segments(in_json)
    ref_words = tokenize_simple(ref_text)

    hyp_words, seg_ranges = build_global_hypothesis_and_boundaries(segments)

    # Edge case: no HYP words at all
    if len(hyp_words) == 0:
        out_segments = []
        for seg in segments:
            hyp = seg.get("sentence", "")
            cer = compute_cer(hyp, "")
            cer_percent = f"{cer * 100:.2f}%"
            out_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "PredictedOut": hyp,
                "ground_truth": "",
                "cer": cer_percent,
            })
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"segments": out_segments}, f, ensure_ascii=False, indent=2)
        return

    alignments = compute_alignment(ref_words, hyp_words)
    hyp_to_ref_start, hyp_to_ref_end = build_hyp_to_ref_map(len(hyp_words), alignments)
    seg_gt_texts = extract_segment_ground_truth(seg_ranges, ref_words, hyp_to_ref_start, hyp_to_ref_end)

    out_segments = []
    for seg, gt in zip(segments, seg_gt_texts):
        hyp = seg.get("sentence", "") or ""
        cer = compute_cer(hyp, gt)
        cer_percent = f"{cer * 100:.2f}%"

        out_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "PredictedOut": hyp,
            "ground_truth": gt,
            "cer": cer_percent,
        })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"segments": out_segments}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
