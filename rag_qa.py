# rag_qa.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# -------------------
# Config
# -------------------
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
TOP_K = int(os.environ.get("TOP_K", "5"))
NORMALIZE = True  # must match how you built the index

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def load_index(index_dir: Path):
    index = faiss.read_index(str(index_dir / "chunks.faiss"))
    metadata = list(read_jsonl(index_dir / "metadata.jsonl"))
    return index, metadata

def load_ingest(ingest_jsonl: Path):
    pages = list(read_jsonl(ingest_jsonl))
    lut = {(p["file"], p["page_index"]): p for p in pages}
    return lut

def to_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Build a grounded prompt with explicit citation instructions.
    """
    header = (
        "You are a careful assistant. Use ONLY the provided context.\n"
        "Cite evidence like [p:PAGE_LABEL] after each sentence.\n"
        "If unsure, say you don't know.\n"
        "Then list an 'Evidence' section with the exact quoted snippets.\n\n"
    )
    ctx_lines = []
    for i, c in enumerate(contexts, 1):
        plabel = c["citation"].get("page_label")
        pdisp = plabel if plabel not in (None, "") else str(c["citation"]["page_index"] + 1)
        ctx_lines.append(
            f"### Context {i} — file={Path(c['citation']['file']).name}, page={pdisp}\n{c['text']}\n"
        )
    ctx = "\n".join(ctx_lines)
    q = f"\nQuestion: {question}\n\nAnswer:"
    return header + ctx + q

def call_ollama(model: str, prompt: str) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

def union_boxes(boxes: List[List[float]]):
    if not boxes: return None
    xs0, ys0, xs1, ys1 = zip(*boxes)
    return [min(xs0), min(ys0), max(xs1), max(ys1)]

def compute_bbox(meta: Dict[str, Any], page: Dict[str, Any]):
    h = meta["highlight"]
    if h["type"] == "native":
        blocks = page.get("blocks", [])
        sel = [blocks[i]["bbox"] for i in h.get("block_indices", []) if 0 <= i < len(blocks)]
        return {"type":"native","bbox":union_boxes(sel), "dpi":None}
    else:
        words = page.get("words", [])
        sel = [words[i]["bbox"] for i in h.get("word_indices", []) if 0 <= i < len(words)]
        return {"type":"ocr","bbox":union_boxes(sel), "dpi":page.get("dpi")}

def retrieve(index, metadata, embed_model, query: str, top_k=TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    if NORMALIZE:
        faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb.astype("float32"), top_k)
    hits = []
    for rank, idx in enumerate(I[0].tolist()):
        m = metadata[idx]
        hits.append({
            "rank": rank+1,
            "score": float(D[0][rank]),
            "text": m["text"],
            "citation": {
                "file": m["file"], "page_index": m["page_index"],
                "page_label": m.get("page_label"), "mode": m["mode"]
            },
            "highlight": m["highlight"]
        })
    return hits

def answer(query: str, index_dir: str, ingest_jsonl: str, model: str = OLLAMA_MODEL):
    index, metadata = load_index(Path(index_dir))
    pages = load_ingest(Path(ingest_jsonl))
    embed = SentenceTransformer(EMBED_MODEL)

    hits = retrieve(index, metadata, embed, query, top_k=TOP_K)
    # attach display-ready bboxes
    viz = []
    for h in hits:
        page = pages[(h["citation"]["file"], h["citation"]["page_index"])]
        viz.append({
            **h,
            "display": compute_bbox(h, page)
        })

    prompt = to_prompt(query, viz)
    completion = call_ollama(model, prompt)
    return {
        "query": query,
        "model": model,
        "answer": completion,
        "results": viz
    }

if __name__ == "__main__":
    import argparse, pprint
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--ingest_jsonl", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--model", default=OLLAMA_MODEL)
    args = ap.parse_args()

    out = answer(args.query, args.index_dir, args.ingest_jsonl, args.model)
    print(json.dumps(out, ensure_ascii=False, indent=2))
