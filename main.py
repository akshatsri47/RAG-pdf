# main.py — FastAPI version of your RAG server
from __future__ import annotations
import io, os, json, uuid, subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from werkzeug.utils import secure_filename

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

# --- import your existing pipeline
from rag_qa import answer  # def answer(query, index_dir, ingest_jsonl, model=...) -> dict

# --------------------------
import sys
# Config / Folders
# --------------------------
BASE_DIR = Path(__file__).parent.resolve()

# Default demo locations (optional)
DEMO_JSON_DIR  = BASE_DIR / "json"
DEMO_JSONL     = DEMO_JSON_DIR / "sample_out.jsonl"
DEMO_INDEX_DIR = BASE_DIR / "rag_index"

UPLOAD_DIR = BASE_DIR / "uploads"
STATE_DIR  = BASE_DIR / "state"
UPLOAD_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

LAST_PATHS = STATE_DIR / "last_paths.json"

DEFAULT_LANG = "eng"   # Tesseract OCR language
NORMALIZE    = True    # must match your build_index usage
DEFAULT_DPI  = 150     # preview DPI
ALLOWED_EXTS = {".pdf"}

# --------------------------
# FastAPI appimport sys
print("Interpreter:", sys.executable)

# --------------------------
app = FastAPI(title="PDF RAG (FastAPI)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Helpers
# --------------------------
def save_current(index_dir: Path, ingest_jsonl: Path, title: str):
    LAST_PATHS.write_text(json.dumps({
        "index_dir": str(index_dir.resolve()),
        "ingest_jsonl": str(ingest_jsonl.resolve()),
        "title": title
    }, ensure_ascii=False), encoding="utf-8")

def load_current() -> Optional[Dict[str, str]]:
    if LAST_PATHS.exists():
        try:
            return json.loads(LAST_PATHS.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def list_recent() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not UPLOAD_DIR.exists():
        return out
    for p in sorted(UPLOAD_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        idx = p / "index"
        j = p / "out.jsonl"
        if idx.exists() and j.exists():
            title = next((f.name for f in p.glob("*.pdf")), p.name)
            out.append({
                "key": p.name,
                "title": title,
                "index_dir": str(idx.resolve()),
                "ingest_jsonl": str(j.resolve())
            })
        if len(out) >= 20:
            break
    return out

def get_active_paths() -> Dict[str, str]:
    cur = load_current()
    if cur:
        if Path(cur["index_dir"]).exists() and Path(cur["ingest_jsonl"]).exists():
            return cur
    recents = list_recent()
    if recents:
        r0 = recents[0]
        save_current(Path(r0["index_dir"]), Path(r0["ingest_jsonl"]), r0["title"])
        return load_current() or {}
    if DEMO_INDEX_DIR.exists() and DEMO_JSONL.exists():
        return {
            "index_dir": str(DEMO_INDEX_DIR.resolve()),
            "ingest_jsonl": str(DEMO_JSONL.resolve()),
            "title": "Demo"
        }
    return {"index_dir": "", "ingest_jsonl": "", "title": ""}

def ensure_pdf_ext(name: str):
    if Path(name).suffix.lower() not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="Only .pdf files are allowed")

def run_ingest_and_index(pdf_path: Path, jsonl_path: Path, index_dir: Path):
    # Use "python" or "py" depending on your Windows setup.
    py_cmd = "python"

    # 1) ingest
    try:
        subprocess.check_call([
            py_cmd, str(BASE_DIR / "ingest_pdf.py"),
            str(pdf_path),
            "--out", str(jsonl_path),
            "--lang", DEFAULT_LANG
        ])
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    # 2) build index
    try:
        cmd = [
            py_cmd, str(BASE_DIR / "build_index.py"),
            "--jsonl", str(jsonl_path),
            "--out_dir", str(index_dir)
        ]
        if NORMALIZE:
            cmd.append("--normalize")
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Index build failed: {e}")

# --------------------------
# Schemas
# --------------------------
class QARequest(BaseModel):
    query: str

class QAResponse(BaseModel):
    query: str
    model: str
    answer: str
    results: List[Dict[str, Any]]

class SwitchRequest(BaseModel):
    key: str

# --------------------------
# Routes
# --------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/status")
def status():
    """Return active paths and recent uploads."""
    return {
        "active": get_active_paths(),
        "recent": list_recent()
    }

@app.post("/upload")
async def upload(pdf: UploadFile = File(...)):
    """Upload a PDF → ingest → index; become the new active dataset."""
    ensure_pdf_ext(pdf.filename)

    uid = str(uuid.uuid4())[:8]
    workdir = UPLOAD_DIR / uid
    workdir.mkdir(parents=True, exist_ok=True)

    fname = secure_filename(pdf.filename)
    pdf_path = workdir / fname
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    jsonl_path = workdir / "out.jsonl"
    index_dir  = workdir / "index"
    run_ingest_and_index(pdf_path, jsonl_path, index_dir)

    save_current(index_dir, jsonl_path, pdf_path.name)

    return {
        "message": "Upload processed",
        "key": uid,
        "title": pdf_path.name,
        "index_dir": str(index_dir.resolve()),
        "ingest_jsonl": str(jsonl_path.resolve())
    }

@app.post("/switch")
def switch(req: SwitchRequest):
    """Switch active dataset to a previous upload by key."""
    workdir = UPLOAD_DIR / req.key
    idx = workdir / "index"
    j = workdir / "out.jsonl"
    if not (idx.exists() and j.exists()):
        raise HTTPException(status_code=404, detail="Upload key not found or incomplete")
    title = next((f.name for f in workdir.glob("*.pdf")), req.key)
    save_current(idx, j, title)
    return {"message": f"Switched to {title}", "index_dir": str(idx), "ingest_jsonl": str(j)}

@app.post("/qa", response_model=QAResponse)
def qa(body: QARequest):
    """Ask a question against the active dataset."""
    current = get_active_paths()
    if not (current["index_dir"] and current["ingest_jsonl"]):
        raise HTTPException(status_code=400, detail="No active dataset. Upload a PDF first.")
    if not (Path(current["index_dir"]).exists() and Path(current["ingest_jsonl"]).exists()):
        raise HTTPException(status_code=400, detail="Active dataset missing. Upload a PDF again.")

    out = answer(body.query, current["index_dir"], current["ingest_jsonl"])
    return out  # matches QAResponse structure

@app.get("/preview")
def preview(
    file: str = Query(..., description="Absolute path to the source PDF"),
    page: int = Query(0, ge=0),
    type: str = Query("native", pattern="^(native|ocr)$"),
    dpi: int = Query(DEFAULT_DPI, ge=100, le=600),
    bbox: Optional[str] = Query(None, description="JSON array [x0,y0,x1,y1] (PDF points for native, pixels for OCR)")
):
    """Render a page to PNG with an optional highlight rectangle."""
    try:
        bbox_vals = json.loads(bbox) if bbox else None
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bbox JSON")

    doc = fitz.open(file)
    if page >= len(doc):
        raise HTTPException(status_code=400, detail="Page index out of range")
    pg = doc[page]

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = pg.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    if bbox_vals:
        draw = ImageDraw.Draw(img, "RGBA")
        if type == "native":
            x0, y0, x1, y1 = bbox_vals
            rect = [x0*zoom, y0*zoom, x1*zoom, y1*zoom]
        else:
            rect = bbox_vals
        draw.rectangle(rect, outline=(255, 0, 0, 255), width=4)
        draw.rectangle(rect, fill=(255, 0, 0, 60))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
