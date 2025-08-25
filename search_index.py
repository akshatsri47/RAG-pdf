# search_index.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def union_boxes(boxes: List[List[float]]) -> List[float] | None:
    if not boxes:
        return None
    xs0, ys0, xs1, ys1 = zip(*boxes)
    return [min(xs0), min(ys0), max(xs1), max(ys1)]

def highlight_box_for_hit(hit: Dict[str, Any], page_payload: Dict[str, Any]) -> Tuple[str, List[float] | None]:
    """
    Return ("native"/"ocr", union_bbox)
    - Native: union of block bboxes (PDF points)
    - OCR: union of word bboxes (pixels at given DPI)
    """
    h = hit["highlight"]
    if h["type"] == "native":
        blocks = page_payload.get("blocks", [])
        sel = [blocks[i]["bbox"] for i in h.get("block_indices", []) if i < len(blocks)]
        return "native", union_boxes(sel)
    else:
        words = page_payload.get("words", [])
        sel = [words[i]["bbox"] for i in h.get("word_indices", []) if i < len(words)]
        return "ocr", union_boxes(sel)

def main():
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True, help="Directory with chunks.faiss and metadata.jsonl")
    ap.add_argument("--ingest_jsonl", required=True, help="Original ingest JSONL (to fetch page boxes)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--query", required=True)
    args = ap.parse_args()

    index = faiss.read_index(str(Path(args.index_dir) / "chunks.faiss"))
    metadata = read_jsonl(Path(args.index_dir) / "metadata.jsonl")
    pages = read_jsonl(Path(args.ingest_jsonl))
    # quick lookup table: (file, page_index) -> page payload
    page_lookup = {(p["file"], p["page_index"]): p for p in pages}

    model = SentenceTransformer(args.model)
    q_emb = model.encode([args.query], convert_to_numpy=True)

    if args.normalize:
        faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb.astype("float32"), args.top_k)
    hits = []
    for rank, idx in enumerate(I[0].tolist()):
        meta = metadata[idx]
        page_payload = page_lookup[(meta["file"], meta["page_index"])]
        htype, bbox = highlight_box_for_hit(meta, page_payload)
        hits.append({
            "rank": rank + 1,
            "score": float(D[0][rank]),
            "text": meta["text"],
            "citation": {
                "file": meta["file"],
                "page_index": meta["page_index"],
                "page_label": meta.get("page_label"),
                "mode": meta["mode"],
            },
            "highlight": {
                "type": htype,            
                "bbox": bbox,
                "dpi": page_payload.get("dpi"),
            }
        })

    print(json.dumps({"query": args.query, "results": hits}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
