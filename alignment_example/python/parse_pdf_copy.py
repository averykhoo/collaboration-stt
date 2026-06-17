#!/usr/bin/env python3
import sys
import re
import json
import argparse
import unicodedata

# ============================================================
# NUMBER LEXICON (Malay)
# ============================================================

ONES = [
    "kosong", "satu", "dua", "tiga", "empat",
    "lima", "enam", "tujuh", "lapan", "sembilan"
]

MONTHS = {
    "01": "januari", "02": "februari", "03": "mac",
    "04": "april", "05": "mei", "06": "jun",
    "07": "julai", "08": "ogos", "09": "september",
    "10": "oktober", "11": "november", "12": "disember"
}


def number_to_malay(n: int) -> str:
    if n < 10:
        return ONES[n]
    if n == 10:
        return "sepuluh"
    if n == 11:
        return "sebelas"
    if n < 20:
        return ONES[n - 10] + " belas"
    if n < 100:
        d, r = divmod(n, 10)
        return ONES[d] + " puluh" if r == 0 else ONES[d] + " puluh " + ONES[r]
    if n == 100:
        return "seratus"
    if n < 200:
        return "seratus " + number_to_malay(n - 100)
    if n < 1000:
        d, r = divmod(n, 100)
        return ONES[d] + " ratus" if r == 0 else ONES[d] + " ratus " + number_to_malay(r)
    if n == 1000:
        return "seribu"
    if n < 2000:
        return "seribu " + number_to_malay(n - 1000)
    if n < 1_000_000:
        d, r = divmod(n, 1000)
        return number_to_malay(d) + " ribu" if r == 0 else number_to_malay(d) + " ribu " + number_to_malay(r)
    if n < 1_000_000_000:
        d, r = divmod(n, 1_000_000)
        return number_to_malay(d) + " juta" if r == 0 else number_to_malay(d) + " juta " + number_to_malay(r)
    d, r = divmod(n, 1_000_000_000)
    return number_to_malay(d) + " bilion" if r == 0 else number_to_malay(d) + " bilion " + number_to_malay(r)


def decimal_to_malay(s: str) -> str:
    left, right = s.split(".")
    return number_to_malay(int(left)) + " perpuluhan " + " ".join(ONES[int(d)] for d in right)


# ============================================================
# UNIT / DATE / RANGE
# ============================================================

def money_to_malay(token: str) -> str:
    s = token.lower().replace("rm", "")
    if "." in s:
        r, sen = s.split(".")
        r = int(r) if r else 0
        sen = int(sen.ljust(2, "0")[:2]) if sen else 0
        out = []
        if r:
            out.append(number_to_malay(r) + " ringgit")
        if sen:
            out.append(number_to_malay(sen) + " sen")
        return " ".join(out)
    return number_to_malay(int(s)) + " ringgit"


def unit_simple(token: str, suffix: str, word: str) -> str:
    n = token[:-len(suffix)]
    return decimal_to_malay(n) + " " + word if "." in n else number_to_malay(int(n)) + " " + word


def percent_to_malay(token: str) -> str:
    return number_to_malay(int(token[:-1])) + " peratus"


def date_to_malay(token: str) -> str:
    d, m, y = token.split("/")
    return f"{number_to_malay(int(d))} {MONTHS[m]} {number_to_malay(int(y))}"


def range_to_malay(token: str) -> str:
    a, b = token.split("-")
    return f"{number_to_malay(int(a))} hingga {number_to_malay(int(b))}"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str, mode: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    text = re.sub(r"\b\d{2}/\d{2}/\d{4}\b", lambda m: date_to_malay(m.group()), text)
    text = re.sub(r"\b\d+-\d+\b", lambda m: range_to_malay(m.group()), text)

    if mode == "gold":
        text = re.sub(r"\brm\d+(?:\.\d+)?\b", lambda m: money_to_malay(m.group()), text)
        text = re.sub(r"\b\d+(?:\.\d+)?km\b", lambda m: unit_simple(m.group(), "km", "kilometer"), text)
        text = re.sub(r"\b\d+(?:\.\d+)?kg\b", lambda m: unit_simple(m.group(), "kg", "kilogram"), text)
        text = re.sub(r"\b\d+\.\d+\b", lambda m: decimal_to_malay(m.group()), text)
        text = re.sub(r"\b\d+%\b", lambda m: percent_to_malay(m.group()), text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b\d+\b", lambda m: number_to_malay(int(m.group())), text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_speaker(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = s.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s)).strip()


# ============================================================
# FILTERS
# ============================================================

def is_attendance_line(line: str) -> bool:
    if ":" in line:
        return False
    if "(" in line and ")" in line:
        if re.search(r"\b(datuk|dato|puan|tuan|haji|seri|dr)\b", line.lower()):
            return 3 <= len(line.split()) <= 12
    return False


SPEAKER_HEADER_RE = re.compile(r"^\s*(.+?)\s*:\s*(.*)$")

SPEAKER_KEYWORDS = [
    "tuan", "puan", "dato", "datuk", "menteri",
    "yang di-pertua", "yang dipertua", "perdana"
]


def parse_speaker_header(line: str):
    if ":" not in line:
        return None
    low = line.lower()
    if not any(k in low for k in SPEAKER_KEYWORDS):
        return None
    m = SPEAKER_HEADER_RE.match(line)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


# ============================================================
# SEGMENTATION (DROP FRONT MATTER)
# ============================================================

def segment(lines):
    segments = []
    current = None
    started = False

    def flush():
        nonlocal current
        if current and current["text"]:
            segments.append(current)
        current = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if not started:
            parsed = parse_speaker_header(line)
            if not parsed:
                continue
            started = True

        if is_attendance_line(line):
            continue

        parsed = parse_speaker_header(line)
        if parsed:
            flush()
            speaker, after = parsed
            current = {
                "speaker_raw": speaker,
                "text": after
            }
            continue

        if current:
            current["text"] += " " + line

    flush()
    return segments


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_txt")
    ap.add_argument("out_json")
    ap.add_argument("--mode", choices=["soft", "gold"], default="gold")
    ap.add_argument("--min_words", type=int, default=3)
    args = ap.parse_args()

    with open(args.in_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    segments = segment(lines)

    with open(args.out_json, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            text_norm = normalize_text(seg["text"], args.mode)
            if len(text_norm.split()) < args.min_words:
                continue

            out = {
                "id": i,
                "speaker_raw": seg["speaker_raw"],
                "speaker_norm": normalize_speaker(seg["speaker_raw"]),
                "text_raw": seg["text"],
                "text_norm": text_norm,
                "mode": args.mode
            }

            json.dump(out, f, ensure_ascii=False, indent=2)
            f.write("\n\n")

    print(f"[DONE] segments: {len(segments)}")
    print(f"[DONE] wrote: {args.out_json}")


if __name__ == "__main__":
    main()
