# app.py — auto paths (no manual inputs), upload → ingest → index → ask
from __future__ import annotations
import io, json, os, uuid, subprocess
from pathlib import Path
from typing import List, Dict, Any
from flask import Flask, request, render_template_string, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
from rag_qa import answer  # reuse pipeline

APP = Flask(__name__)
APP.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
APP.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
ALLOWED_EXTS = {".pdf"}

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
STATE_DIR = BASE_DIR / "state"
UPLOAD_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)
LAST_PATHS = STATE_DIR / "last_paths.json"
DEMO_JSON_DIR  = BASE_DIR / "json"         # <project>/json
DEMO_JSONL     = DEMO_JSON_DIR / "sample_out.jsonl"
DEMO_INDEX_DIR = BASE_DIR / "rag_index"  

DEFAULT_LANG = "eng"    # Tesseract language
NORMALIZE = True
DEFAULT_DPI = 150

TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>RAG over PDFs (local)</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; }
    .row { display: flex; gap: 24px; align-items: flex-start;}
    .col { flex: 1; }
    img { max-width: 100%; height: auto; border: 1px solid #ccc; }
    .hit { margin-bottom: 18px; padding: 8px 12px; border-left: 4px solid #888; background: #f8f8f8; }
    .flash { background:#fff3cd; color:#664d03; padding:8px 12px; border:1px solid #ffe69c; border-radius:6px; margin:10px 0; }
    .box { border:1px solid #e0e0e0; padding:12px; border-radius:6px; margin:12px 0; }
    .muted { color:#666; font-size: 12px; }
    .list li { margin: 6px 0; }
    code { white-space: pre-wrap; }
  </style>
</head>
<body>
  <h2>RAG over PDFs (local stack)</h2>

  {% with messages = get_flashed_messages() %}
    {% if messages %}
      {% for m in messages %}
        <div class="flash">{{ m }}</div>
      {% endfor %}
    {% endif %}
  {% endwith %}

  <div class="row">
    <div class="col" style="max-width:520px;">
      <div class="box">
        <form method="POST" action="{{ url_for('upload') }}" enctype="multipart/form-data">
          <strong>Upload a PDF</strong><br/>
          <input type="file" name="pdf" accept="application/pdf" required />
          <button type="submit">Upload & Process</button>
          <div class="muted">Ingests & builds a fresh index locally.</div>
        </form>
      </div>

      <div class="box">
        <form method="GET" action="/">
          <strong>Ask a question</strong><br/>
          <input type="text" name="q" placeholder="Type your question…" value="{{q or ''}}" size="60"/>
          <button type="submit">Ask</button>
          <div class="muted">
            Using: <code>{{ current.index_dir }}</code><br/>
            Ingest JSONL: <code>{{ current.ingest_jsonl }}</code>
          </div>
        </form>
      </div>

      {% if recent and recent|length > 0 %}
      <div class="box">
        <strong>Recent uploads</strong>
        <ul class="list">
          {% for r in recent %}
            <li>
              {{ r.title }} —
              <a href="{{ url_for('switch', key=r.key) }}">Use this</a>
              <div class="muted">{{ r.index_dir }}</div>
            </li>
          {% endfor %}
        </ul>
      </div>
      {% endif %}
    </div>

    <div class="col">
      {% if resp %}
        <h3>Answer</h3>
        <div style="white-space: pre-wrap;">{{ resp.answer }}</div>
        <h3>Top passages</h3>
        {% for r in resp.results %}
          <div class="hit">
            <div><strong>{{ loop.index }}.</strong> {{ r.citation.file | basename }} — page {{ (r.citation.page_label if r.citation.page_label else (r.citation.page_index + 1)) }}</div>
            <div class="muted">score: {{ "%.4f"|format(r.score) }}</div>
            <div style="margin-top:6px; white-space: pre-wrap;">{{ r.text }}</div>
            <div style="margin-top:6px;">
              <img src="/preview?file={{ r.citation.file|urlencode }}&page={{ r.citation.page_index }}&type={{ r.display.type }}&dpi={{ r.display.dpi or """ + str(DEFAULT_DPI) + """ }}&bbox={{ r.display.bbox|tojson | urlencode }}" />
            </div>
          </div>
        {% endfor %}
      {% else %}
        <div class="muted">Upload a PDF and ask something to see results here.</div>
      {% endif %}

      {% if resp %}
      <h3>JSON (debug)</h3>
      <code>{{ resp | tojson(indent=2) }}</code>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""

@APP.template_filter('basename')
def basename_filter(p): return Path(p).name

# ---------- state helpers ----------
def save_current(index_dir: Path, ingest_jsonl: Path, title: str):
    LAST_PATHS.write_text(json.dumps({
        "index_dir": str(index_dir.resolve()),
        "ingest_jsonl": str(ingest_jsonl.resolve()),
        "title": title
    }, ensure_ascii=False), encoding="utf-8")

def load_current() -> Dict[str, str] | None:
    if LAST_PATHS.exists():
        try:
            return json.loads(LAST_PATHS.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def list_recent() -> List[Dict[str, str]]:
    out = []
    if not UPLOAD_DIR.exists():
        return out
    # each upload: uploads/<key>/ with 'index/' and 'out.jsonl'
    for p in sorted(UPLOAD_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        idx = p / "index"
        j = p / "out.jsonl"
        if idx.exists() and j.exists():
            out.append({
                "key": p.name,
                "index_dir": str(idx.resolve()),
                "ingest_jsonl": str(j.resolve()),
                "title": next((f.name for f in p.glob("*.pdf")), p.name)
            })
        if len(out) >= 10:
            break
    return out
def get_default_paths():
    # 1) use remembered "current" if both paths still exist
    cur = load_current()
    if cur:
        idx_ok = Path(cur["index_dir"]).exists()
        jsonl_ok = Path(cur["ingest_jsonl"]).exists()
        if idx_ok and jsonl_ok:
            return cur

    # 2) otherwise prefer the most recent upload (if any)
    recent = list_recent()
    if recent:
        r0 = recent[0]
        save_current(Path(r0["index_dir"]), Path(r0["ingest_jsonl"]), r0["title"])
        return load_current()

    # 3) fallback: demo paths in your repo (json/sample_out.jsonl + rag_index/)
    if DEMO_INDEX_DIR.exists() and DEMO_JSONL.exists():
        return {
            "index_dir": str(DEMO_INDEX_DIR.resolve()),
            "ingest_jsonl": str(DEMO_JSONL.resolve()),
            "title": "Demo"
        }

    # 4) nothing available yet (prompt user to upload)
    return {"index_dir": "", "ingest_jsonl": "", "title": ""}


# ---------- routes ----------
@APP.route("/", methods=["GET"])
def home():
    q = request.args.get("q")
    current = get_default_paths()
    resp = None
    if q:
        resp = answer(q, current["index_dir"], current["ingest_jsonl"])
        class O(dict): __getattr__ = dict.get
        resp = O(resp)
        resp.results = [O({**r, "citation": O(r["citation"]), "display": O(r["display"])}) for r in resp.results]
    return render_template_string(TEMPLATE, q=q, current=current, resp=resp, recent=list_recent())

@APP.route("/switch/<key>")
def switch(key: str):
    # switch to a recent upload by key
    workdir = UPLOAD_DIR / key
    idx = workdir / "index"
    j = workdir / "out.jsonl"
    if idx.exists() and j.exists():
        title = next((f.name for f in workdir.glob("*.pdf")), key)
        save_current(idx, j, title)
        flash(f"Switched to {title}")
    else:
        flash("Selected upload is incomplete.")
    return redirect(url_for("home"))

@APP.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("pdf")
    if not file or file.filename == "":
        flash("No file selected.")
        return redirect(url_for("home"))

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        flash("Only PDF files are allowed.")
        return redirect(url_for("home"))

    uid = str(uuid.uuid4())[:8]
    workdir = UPLOAD_DIR / uid
    workdir.mkdir(parents=True, exist_ok=True)

    # save pdf
    fname = secure_filename(file.filename)
    pdf_path = workdir / fname
    file.save(pdf_path)

    # ingest
    jsonl_path = workdir / "out.jsonl"
    try:
        flash("Ingesting PDF…")
        subprocess.check_call([
            "python", str(BASE_DIR / "ingest_pdf.py"),
            str(pdf_path),
            "--out", str(jsonl_path),
            "--lang", DEFAULT_LANG
        ])
    except subprocess.CalledProcessError as e:
        flash(f"Ingestion failed: {e}")
        return redirect(url_for("home"))

    # index
    index_dir = workdir / "index"
    try:
        flash("Building index…")
        cmd = [
            "python", str(BASE_DIR / "build_index.py"),
            "--jsonl", str(jsonl_path),
            "--out_dir", str(index_dir)
        ]
        if NORMALIZE:
            cmd.append("--normalize")
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        flash(f"Index build failed: {e}")
        return redirect(url_for("home"))

    # remember current
    save_current(index_dir, jsonl_path, pdf_path.name)
    flash(f"Ready: {pdf_path.name}")
    return redirect(url_for("home"))

@APP.route("/preview")
def preview():
    current = get_default_paths()
    file = request.args.get("file")
    page_i = int(request.args.get("page", "0"))
    typ = request.args.get("type", "native")
    dpi = int(request.args.get("dpi", str(DEFAULT_DPI)))
    bbox_json = request.args.get("bbox")
    bbox = json.loads(bbox_json) if bbox_json and bbox_json != "null" else None

    doc = fitz.open(file)
    page = doc[page_i]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    if bbox:
        draw = ImageDraw.Draw(img, "RGBA")
        if typ == "native":
            x0, y0, x1, y1 = bbox
            rect = [x0*zoom, y0*zoom, x1*zoom, y1*zoom]
        else:
            rect = bbox
        draw.rectangle(rect, outline=(255,0,0,255), width=4)
        draw.rectangle(rect, fill=(255,0,0,60))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    APP.run(host="127.0.0.1", port=5000, debug=True)
