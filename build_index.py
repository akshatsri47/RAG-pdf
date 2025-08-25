# build_index.py
from __future__ import annotations
import os, json, math
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss

# ----------------------------
# config
# ----------------------------
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 800         # chars per chunk
CHUNK_OVERLAP = 150      # chars overlap
MAX_PAGES = None         # set e.g. 50 while testing

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def simple_chunks(text: str, chunk_size: int, overlap: int):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        chunks.append((start, end, chunk))
        if end == n:
            break
        start = max(end - overlap, 0)
    return chunks

def build_records(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize pages to chunk records with provenance for highlighting.
    Each record has:
      - text
      - file, page_index, mode
      - highlight: {type: 'native'|'ocr', indices: [i0,i1,...]}
    """
    records = []
    for p in pages[: (MAX_PAGES or len(pages))]:
        base = {
            "file": p["file"],
            "page_index": p["page_index"],
            "page_label": p.get("page_label"),
            "mode": p["mode"],
            "width": p["width"],
            "height": p["height"],
        }

        if p["mode"] == "native":
            
            blocks = p.get("blocks", [])
            pieces = []
            block_spans = []  
            cursor = 0
            for i, b in enumerate(blocks):
                t = (b.get("text") or "").strip()
                if not t:
                    continue
                if pieces:
                    pieces.append("\n")
                    cursor += 1
                pieces.append(t)
                start = cursor
                cursor += len(t)
                end = cursor
                block_spans.append((start, end, i))
            full = "".join(pieces)

            for s, e, chunk in simple_chunks(full, CHUNK_SIZE, CHUNK_OVERLAP):
                
                block_ids = sorted({idx for (bs, be, idx) in block_spans if not (be <= s or bs >= e)})
                rec = {
                    **base,
                    "text": chunk,
                    "highlight": {
                        "type": "native",
                        "block_indices": block_ids,
                    },
                }
                records.append(rec)

        else:
            # OCR path: join words with spaces; capture word spans
            words = p.get("words", [])
            tokens = []
            word_spans = []  
            cursor = 0
            for i, w in enumerate(words):
                t = (w.get("text") or "").strip()
                if not t:
                    continue
                if tokens:
                    tokens.append(" ")
                    cursor += 1
                tokens.append(t)
                start = cursor
                cursor += len(t)
                end = cursor
                word_spans.append((start, end, i))

            full = "".join(tokens)

            for s, e, chunk in simple_chunks(full, CHUNK_SIZE, CHUNK_OVERLAP):
                word_ids = sorted({idx for (ws, we, idx) in word_spans if not (we <= s or ws >= e)})
                rec = {
                    **base,
                    "text": chunk,
                    "highlight": {
                        "type": "ocr",
                        "word_indices": word_ids,
                        "dpi": pages[0].get("dpi") or p.get("dpi"),  # carry through if present
                    },
                }
                records.append(rec)

    return records

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to ingest output JSONL")
    ap.add_argument("--out_dir", default="rag_index", help="Output folder")
    ap.add_argument("--model", default=EMBED_MODEL, help="SentenceTransformer model name")
    ap.add_argument("--dim", type=int, default=None, help="Override embedding dim (usually auto)")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize embeddings")
    args = ap.parse_args()

    in_path = Path(args.jsonl).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {in_path}")
    pages = read_jsonl(in_path)

    print("[chunk] building records…")
    records = build_records(pages)
    texts = [r["text"] for r in records]
    print(f"[chunk] {len(records)} chunks")

    print(f"[embed] model={args.model}")
    model = SentenceTransformer(args.model)
    embs = model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True)

    if args.normalize:
        faiss_normalize_L2 = faiss.normalize_L2
        embs = embs.astype("float32")
        faiss_normalize_L2(embs)

    dim = args.dim or embs.shape[1]
    index = faiss.IndexFlatIP(dim) if args.normalize else faiss.IndexFlatL2(dim)
    index.add(embs.astype("float32"))

    # save index + metadata
    faiss.write_index(index, str(out_dir / "chunks.faiss"))
    with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {out_dir/'chunks.faiss'} and {out_dir/'metadata.jsonl'}")

if __name__ == "__main__":
    main()
