"""
ingest_pdf.py — Windows-friendly PDF detector & extractor
- Detect native vs scanned pages
- Extract text + evidence boxes (bboxes) for both paths
- Output JSONL (one object per page) for downstream RAG

Run:
  python ingest_pdf.py "C:\path\to\file.pdf" --out out.jsonl --lang eng --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"
"""

from __future__ import annotations
import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
from PIL import Image
import pytesseract


def configure_tesseract(tesseract_path: str | None):
    """
    Set Tesseract executable path on Windows if not on PATH.
    """
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path


def page_has_native_text(page: fitz.Page, min_chars: int = 40) -> bool:
    """
    Heuristic to classify a page as native (true) vs scanned (false).
    - Checks extracted text length and presence of 'text' blocks.
    """
    txt = page.get_text("text") or ""
    if len(txt.strip()) >= min_chars:
        return True

    # Double-check via block analysis (robust to very short pages)
    blocks = page.get_text("blocks") or []
    text_blocks = [b for b in blocks if len(b) >= 6 and isinstance(b[4], str) and b[4].strip()]
    return len(text_blocks) > 0


def extract_native(page: fitz.Page) -> Dict[str, Any]:
    """
    Extract text + block-level bboxes from native PDF text.
    Returns:
        {
          "mode": "native",
          "text": "...",
          "blocks": [
             {"bbox": [x0,y0,x1,y1], "text": "..."}
          ]
        }
    """
    text = page.get_text("text") or ""
    blocks = []
    for b in page.get_text("blocks") or []:
        # PyMuPDF 'blocks' entries: (x0, y0, x1, y1, text, block_no, block_type, ...)
        if len(b) >= 6:
            x0, y0, x1, y1, t = b[0], b[1], b[2], b[3], b[4]
            if isinstance(t, str) and t.strip():
                blocks.append({"bbox": [x0, y0, x1, y1], "text": t.strip()})
    return {"mode": "native", "text": text.strip(), "blocks": blocks}


def extract_ocr(page: fitz.Page, dpi: int = 300, lang: str = "eng") -> Dict[str, Any]:
    """
    Render a page to a bitmap and OCR with Tesseract.
    Returns:
        {
          "mode": "ocr",
          "text": "...",
          "words": [
             {"bbox": [x0,y0,x1,y1], "text": "word", "conf": 89}
          ]
        }
    """
    # Render page to image (300 DPI recommended for Tesseract)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # OCR with per-word boxes; psm 6 = assume a block of text
    ocr_data = pytesseract.image_to_data(
        img,
        lang=lang,
        config="--oem 1 --psm 6",
        output_type=pytesseract.Output.DICT
    )

    words = []
    acc_text_tokens = []
    n = len(ocr_data.get("text", []))
    for i in range(n):
        w = (ocr_data["text"][i] or "").strip()
        conf_str = ocr_data["conf"][i]
        try:
            conf = float(conf_str)
        except Exception:
            conf = -1.0

        if w and conf >= 0:  # Tesseract returns -1 for non-words
            x, y, w_px, h_px = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
            words.append({
                "bbox": [x, y, x + w_px, y + h_px],  # pixel-space at the rendered dpi
                "text": w,
                "conf": conf
            })
            acc_text_tokens.append(w)

    text = " ".join(acc_text_tokens).strip()
    return {"mode": "ocr", "text": text, "words": words, "dpi": dpi}


def process_pdf(pdf_path: Path, out_path: Path, lang: str, dpi: int, tesseract_path: str | None):
    configure_tesseract(tesseract_path)

    doc = fitz.open(pdf_path)
    results: List[Dict[str, Any]] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        is_native = page_has_native_text(page)

        if is_native:
            payload = extract_native(page)
        else:
            payload = extract_ocr(page, dpi=dpi, lang=lang)

        # Normalize structure and add page metadata
        page_obj = {
            "file": str(pdf_path),
            "page_index": page_num,                 # 0-based
            "page_label": doc[page_num].get_label(),# respects PDF page labels if present
            "width": page.rect.width,
            "height": page.rect.height,
            **payload
        }
        results.append(page_obj)

    # Write JSONL (one page per line)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] Processed {len(doc)} pages → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect & extract text from PDFs (native + scanned) on Windows using PyMuPDF + Tesseract.")
    parser.add_argument("pdf", type=str, help="Path to PDF file")
    parser.add_argument("--out", type=str, default="out.jsonl", help="Output JSONL path")
    parser.add_argument("--lang", type=str, default="eng", help="Tesseract language(s), e.g. 'eng', 'eng+deu'")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for OCR rendering")
    parser.add_argument("--tesseract", type=str, default=None, help=r"Full path to tesseract.exe if not on PATH, e.g. 'C:\Program Files\Tesseract-OCR\tesseract.exe'")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not pdf_path.exists():
        print(f"[ERR] PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    process_pdf(pdf_path, out_path, lang=args.lang, dpi=args.dpi, tesseract_path=args.tesseract)


if __name__ == "__main__":
    main()
