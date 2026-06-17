#!/usr/bin/env python3

import sys
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

def extract_pdf_text(pdf_path):
    all_pages_text = []

    for page_num, page_layout in enumerate(extract_pages(pdf_path), start=1):
        page_text_blocks = []

        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text()
                if text.strip():
                    page_text_blocks.append(text)

        page_text = "".join(page_text_blocks)
        all_pages_text.append(page_text)

        print(f"[INFO] Page {page_num}: {len(page_text.split())} words")

    return "\n\n".join(all_pages_text)

def main():
    if len(sys.argv) != 3:
        print("Usage: python pdf2txt.py in_pdf out_txt")
        sys.exit(1)

    in_pdf = sys.argv[1]
    out_txt = sys.argv[2]

    print(f"[INFO] Reading PDF: {in_pdf}")
    text = extract_pdf_text(in_pdf)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)

    total_words = len(text.split())
    print(f"[DONE] Written to: {out_txt}")
    print(f"[DONE] Total words extracted: {total_words}")

if __name__ == "__main__":
    main()
