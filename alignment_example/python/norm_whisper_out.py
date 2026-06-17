#!/usr/bin/env python3
import sys
import re
import json
import unicodedata

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


def money_to_malay(token: str) -> str:
    # token: rm123 or rm123.45
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
        return " ".join(out) if out else "kosong ringgit"
    return number_to_malay(int(s)) + " ringgit"


def unit_simple(token: str, suffix: str, word: str) -> str:
    # token: 12kg / 2.5km
    n = token[:-len(suffix)]
    return decimal_to_malay(n) + " " + word if "." in n else number_to_malay(int(n)) + " " + word


def percent_to_malay(token: str) -> str:
    # 15%
    return number_to_malay(int(token[:-1])) + " peratus"


def date_to_malay(token: str) -> str:
    # dd/mm/yyyy
    d, m, y = token.split("/")
    m_word = MONTHS.get(m, m)
    return f"{number_to_malay(int(d))} {m_word} {number_to_malay(int(y))}"


def range_to_malay(token: str) -> str:
    # 10-15 (integers)
    a, b = token.split("-")
    return f"{number_to_malay(int(a))} hingga {number_to_malay(int(b))}"


def normalize_sentence(text: str, mode: str = "gold") -> str:
    """
    Matches PDF gold normalization:
      - lowercase
      - NFKD + ASCII
      - remove thousand separators (30,000 -> 30000)
      - expand dates dd/mm/yyyy
      - expand integer ranges a-b
      - (gold) expand RM, km/kg, decimals, percent
      - remove punctuation to spaces
      - integers -> malay words
      - collapse whitespace
    """
    text = text.lower()

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # remove thousand separators
    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    # dates first
    text = re.sub(r"\b\d{2}/\d{2}/\d{4}\b", lambda m: date_to_malay(m.group()), text)

    # ranges like 10-15 (digits on both sides)
    text = re.sub(r"\b\d+-\d+\b", lambda m: range_to_malay(m.group()), text)

    if mode == "gold":
        # RM
        text = re.sub(r"\brm\d+(?:\.\d+)?\b", lambda m: money_to_malay(m.group()), text)
        # units no-space
        text = re.sub(r"\b\d+(?:\.\d+)?km\b", lambda m: unit_simple(m.group(), "km", "kilometer"), text)
        text = re.sub(r"\b\d+(?:\.\d+)?kg\b", lambda m: unit_simple(m.group(), "kg", "kilogram"), text)
        # decimals
        text = re.sub(r"\b\d+\.\d+\b", lambda m: decimal_to_malay(m.group()), text)
        # percent
        text = re.sub(r"\b\d+%\b", lambda m: percent_to_malay(m.group()), text)

    # final cleanup: keep letters/digits/space only
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # integers -> words
    text = re.sub(r"\b\d+\b", lambda m: number_to_malay(int(m.group())), text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_whisper_file(path: str, mode: str = "gold"):
    segments = []
    ts_re = re.compile(r"\[(\d+\.\d+)\s+(\d+\.\d+)\]\s+(.*)")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = ts_re.match(line)
            if not m:
                continue

            start = float(m.group(1))
            end = float(m.group(2))
            sentence_raw = m.group(3)

            sentence = normalize_sentence(sentence_raw, mode=mode)
            if not sentence:
                continue

            segments.append({
                "start": start,
                "end": end,
                "sentence": sentence
            })

    return segments


def main():
    if len(sys.argv) not in (3, 5):
        print("usage: python norm_whisper_out.py in_txt out.json [--mode gold|soft]")
        sys.exit(1)

    in_txt = sys.argv[1]
    out_json = sys.argv[2]
    mode = "gold"

    if len(sys.argv) == 5:
        if sys.argv[3] != "--mode":
            print("usage: python norm_whisper_out.py in_txt out.json [--mode gold|soft]")
            sys.exit(1)
        mode = sys.argv[4]
        if mode not in ("gold", "soft"):
            print("mode must be gold or soft")
            sys.exit(1)

    segments = parse_whisper_file(in_txt, mode=mode)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, ensure_ascii=False, indent=4)

    print(f"[DONE] segments: {len(segments)}")
    print(f"[DONE] wrote: {out_json}")


if __name__ == "__main__":
    main()
