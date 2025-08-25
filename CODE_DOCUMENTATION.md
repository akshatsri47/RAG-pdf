# RAG Pipeline - Code Documentation

## Table of Contents
1. [PDF Ingestion (`ingest_Pdf.py`)](#pdf-ingestion)
2. [Document Indexing (`build_index.py`)](#document-indexing)
3. [Vector Search (`search_index.py`)](#vector-search)
4. [RAG Engine (`rag_qa.py`)](#rag-engine)
5. [Flask Web Interface (`app.py`)](#flask-web-interface)
6. [FastAPI Web Interface (`main.py`)](#fastapi-web-interface)

---

## PDF Ingestion

### Module: `ingest_Pdf.py`

**Purpose**: Extracts text and metadata from PDF documents with automatic detection of native vs scanned content.

#### Key Functions

##### `configure_tesseract(tesseract_path: str | None)`
Configures the Tesseract OCR executable path, particularly important for Windows systems.

```python
# Example usage
configure_tesseract(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
```

**Parameters**:
- `tesseract_path`: Full path to tesseract executable or None to use system PATH

##### `page_has_native_text(page: fitz.Page, min_chars: int = 40) -> bool`
Heuristic function to determine if a PDF page contains native text or requires OCR.

```python
# Example usage
is_native = page_has_native_text(page, min_chars=50)
```

**Parameters**:
- `page`: PyMuPDF page object
- `min_chars`: Minimum character threshold for native text detection

**Returns**: `True` if page has native text, `False` if OCR is needed

**Algorithm**:
1. Extract text using PyMuPDF's text extraction
2. Check if extracted text length exceeds threshold
3. Fallback: analyze text blocks for robust detection

##### `extract_native(page: fitz.Page) -> Dict[str, Any]`
Extracts text and block-level bounding boxes from native PDF pages.

```python
# Example usage
result = extract_native(page)
# Returns: {
#   "mode": "native",
#   "text": "extracted text content",
#   "blocks": [{"bbox": [x0,y0,x1,y1], "text": "..."}]
# }
```

**Parameters**:
- `page`: PyMuPDF page object

**Returns**: Dictionary with native text extraction results

**Features**:
- Preserves text block structure
- Maintains spatial positioning (bounding boxes)
- Handles mixed content (text + images)

##### `extract_ocr(page: fitz.Page, dpi: int = 300, lang: str = "eng") -> Dict[str, Any]`
Performs OCR on scanned PDF pages using Tesseract.

```python
# Example usage
result = extract_ocr(page, dpi=300, lang="eng")
# Returns: {
#   "mode": "ocr",
#   "text": "OCR extracted text",
#   "words": [{"bbox": [x0,y0,x1,y1], "text": "word", "conf": 89}],
#   "dpi": 300
# }
```

**Parameters**:
- `page`: PyMuPDF page object
- `dpi`: Resolution for image rendering (default: 300)
- `lang`: Tesseract language code (default: "eng")

**Returns**: Dictionary with OCR results including word-level bounding boxes

**Features**:
- High-resolution image rendering for better OCR accuracy
- Word-level confidence scores
- Pixel-space bounding boxes at specified DPI

##### `process_pdf(pdf_path: Path, out_path: Path, lang: str, dpi: int, tesseract_path: str | None)`
Main processing function that orchestrates PDF ingestion.

```python
# Example usage
process_pdf(
    pdf_path=Path("document.pdf"),
    out_path=Path("output.jsonl"),
    lang="eng",
    dpi=300,
    tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
```

**Parameters**:
- `pdf_path`: Path to input PDF file
- `out_path`: Path for output JSONL file
- `lang`: OCR language
- `dpi`: OCR rendering resolution
- `tesseract_path`: Tesseract executable path

**Output Format**:
```json
{
  "file": "path/to/document.pdf",
  "page_index": 0,
  "page_label": "1",
  "width": 612.0,
  "height": 792.0,
  "mode": "native|ocr",
  "text": "extracted text content",
  "blocks": [...],  // native mode
  "words": [...],   // OCR mode
  "dpi": 300        // OCR mode only
}
```

---

## Document Indexing

### Module: `build_index.py`

**Purpose**: Creates searchable vector embeddings from ingested documents with intelligent chunking and highlighting metadata.

#### Key Functions

##### `simple_chunks(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, int, str]]`
Splits text into overlapping chunks for optimal retrieval.

```python
# Example usage
chunks = simple_chunks("long text content", chunk_size=800, overlap=150)
# Returns: [(0, 800, "chunk1"), (650, 1450, "chunk2"), ...]
```

**Parameters**:
- `text`: Input text to chunk
- `chunk_size`: Maximum characters per chunk
- `overlap`: Character overlap between chunks

**Returns**: List of (start, end, chunk_text) tuples

**Algorithm**:
1. Start from beginning of text
2. Create chunk of specified size
3. Move start position by (chunk_size - overlap)
4. Repeat until end of text

##### `build_records(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]`
Creates chunk records with provenance and highlighting metadata.

```python
# Example usage
records = build_records(pages)
# Returns: List of chunk records with highlighting metadata
```

**Parameters**:
- `pages`: List of page objects from PDF ingestion

**Returns**: List of chunk records ready for embedding

**Record Structure**:
```json
{
  "file": "document.pdf",
  "page_index": 0,
  "page_label": "1",
  "mode": "native|ocr",
  "width": 612.0,
  "height": 792.0,
  "text": "chunk text content",
  "highlight": {
    "type": "native|ocr",
    "block_indices": [0, 1, 2],  // native mode
    "word_indices": [0, 1, 2],   // OCR mode
    "dpi": 300                   // OCR mode only
  }
}
```

**Features**:
- Preserves document structure and metadata
- Maintains highlighting information for visual citations
- Handles both native and OCR content types
- Configurable chunking parameters

##### `main()`
Command-line interface for building document indices.

```bash
# Example usage
python build_index.py --jsonl input.jsonl --out_dir rag_index --normalize
```

**Command-line Arguments**:
- `--jsonl`: Path to input JSONL file
- `--out_dir`: Output directory for index files
- `--model`: SentenceTransformer model name
- `--dim`: Override embedding dimension
- `--normalize`: Enable L2 normalization

**Output Files**:
- `chunks.faiss`: FAISS vector index
- `metadata.jsonl`: Chunk metadata with highlighting information

---

## Vector Search

### Module: `search_index.py`

**Purpose**: Performs semantic search over indexed documents with visual highlighting support.

#### Key Functions

##### `union_boxes(boxes: List[List[float]]) -> List[float] | None`
Computes the union bounding box from multiple boxes.

```python
# Example usage
bbox = union_boxes([[0, 0, 10, 10], [5, 5, 15, 15]])
# Returns: [0, 0, 15, 15]
```

**Parameters**:
- `boxes`: List of bounding boxes as [x0, y0, x1, y1]

**Returns**: Union bounding box or None if input is empty

##### `highlight_box_for_hit(hit: Dict[str, Any], page_payload: Dict[str, Any]) -> Tuple[str, List[float] | None]`
Computes display-ready bounding boxes for search results.

```python
# Example usage
highlight_type, bbox = highlight_box_for_hit(hit, page_payload)
# Returns: ("native", [x0, y0, x1, y1]) or ("ocr", [x0, y0, x1, y1])
```

**Parameters**:
- `hit`: Search result with highlighting metadata
- `page_payload`: Original page data from ingestion

**Returns**: Tuple of (highlight_type, bounding_box)

**Features**:
- Handles both native and OCR highlighting
- Converts block/word indices to bounding boxes
- Preserves coordinate spaces (PDF points vs pixels)

##### `main()`
Command-line interface for semantic search.

```bash
# Example usage
python search_index.py --index_dir rag_index --ingest_jsonl input.jsonl --query "search query" --top_k 5
```

**Command-line Arguments**:
- `--index_dir`: Directory containing FAISS index
- `--ingest_jsonl`: Original ingestion JSONL for highlighting
- `--model`: SentenceTransformer model name
- `--normalize`: Enable L2 normalization
- `--top_k`: Number of results to return
- `--query`: Search query

**Output Format**:
```json
{
  "query": "search query",
  "results": [
    {
      "rank": 1,
      "score": 0.85,
      "text": "retrieved text chunk",
      "citation": {
        "file": "document.pdf",
        "page_index": 0,
        "page_label": "1",
        "mode": "native"
      },
      "highlight": {
        "type": "native",
        "bbox": [x0, y0, x1, y1],
        "dpi": null
      }
    }
  ]
}
```

---

## RAG Engine

### Module: `rag_qa.py`

**Purpose**: Orchestrates the complete RAG pipeline from query to answer with local LLM integration.

#### Key Functions

##### `read_jsonl(path: Path)`
Generator function for reading JSONL files line by line.

```python
# Example usage
for item in read_jsonl(Path("data.jsonl")):
    print(item)
```

**Parameters**:
- `path`: Path to JSONL file

**Yields**: Parsed JSON objects

##### `load_index(index_dir: Path) -> Tuple[faiss.Index, List[Dict[str, Any]]]`
Loads FAISS index and metadata from disk.

```python
# Example usage
index, metadata = load_index(Path("rag_index"))
```

**Parameters**:
- `index_dir`: Directory containing index files

**Returns**: Tuple of (FAISS index, metadata list)

##### `load_ingest(ingest_jsonl: Path) -> Dict[Tuple[str, int], Dict[str, Any]]`
Creates lookup table for page data.

```python
# Example usage
page_lookup = load_ingest(Path("input.jsonl"))
# Returns: {(file_path, page_index): page_data}
```

**Parameters**:
- `ingest_jsonl`: Path to ingestion JSONL file

**Returns**: Dictionary mapping (file, page_index) to page data

##### `to_prompt(question: str, contexts: List[Dict[str, Any]]) -> str`
Constructs grounded prompts with explicit citation instructions.

```python
# Example usage
prompt = to_prompt("What is the main topic?", contexts)
```

**Parameters**:
- `question`: User's question
- `contexts`: Retrieved document chunks

**Returns**: Formatted prompt for LLM

**Prompt Structure**:
```
You are a careful assistant. Use ONLY the provided context.
Cite evidence like [p:PAGE_LABEL] after each sentence.
If unsure, say you don't know.
Then list an 'Evidence' section with the exact quoted snippets.

### Context 1 — file=document.pdf, page=1
[context text]

Question: [user question]

Answer:
```

##### `call_ollama(model: str, prompt: str) -> str`
Makes inference requests to local Ollama LLM.

```python
# Example usage
response = call_ollama("llama3.1:8b", prompt)
```

**Parameters**:
- `model`: Ollama model name
- `prompt`: Formatted prompt

**Returns**: LLM response text

**Configuration**:
- Uses environment variable `OLLAMA_URL` (default: `http://localhost:11434/api/generate`)
- 120-second timeout
- Non-streaming mode

##### `union_boxes(boxes: List[List[float]]) -> List[float] | None`
Computes union bounding box (same as search_index.py).

##### `compute_bbox(meta: Dict[str, Any], page: Dict[str, Any]) -> Dict[str, Any]`
Generates display-ready bounding boxes for visual highlighting.

```python
# Example usage
display_info = compute_bbox(metadata, page_data)
# Returns: {"type": "native|ocr", "bbox": [x0,y0,x1,y1], "dpi": 300}
```

**Parameters**:
- `meta`: Chunk metadata with highlighting information
- `page`: Original page data

**Returns**: Display configuration for visual highlighting

##### `retrieve(index, metadata, embed_model, query: str, top_k=TOP_K) -> List[Dict[str, Any]]`
Performs semantic retrieval over indexed documents.

```python
# Example usage
hits = retrieve(index, metadata, embed_model, "search query", top_k=5)
```

**Parameters**:
- `index`: FAISS index
- `metadata`: Chunk metadata
- `embed_model`: SentenceTransformer model
- `query`: Search query
- `top_k`: Number of results

**Returns**: List of search results with scores and metadata

##### `answer(query: str, index_dir: str, ingest_jsonl: str, model: str = OLLAMA_MODEL) -> Dict[str, Any]`
Main RAG function that orchestrates the complete pipeline.

```python
# Example usage
result = answer("What is the main topic?", "rag_index", "input.jsonl", "llama3.1:8b")
```

**Parameters**:
- `query`: User's question
- `index_dir`: Path to FAISS index directory
- `ingest_jsonl`: Path to ingestion JSONL file
- `model`: Ollama model name

**Returns**: Complete RAG response with answer and citations

**Response Format**:
```json
{
  "query": "user question",
  "model": "llama3.1:8b",
  "answer": "LLM generated answer with citations",
  "results": [
    {
      "rank": 1,
      "score": 0.85,
      "text": "retrieved text",
      "citation": {
        "file": "document.pdf",
        "page_index": 0,
        "page_label": "1",
        "mode": "native"
      },
      "highlight": {...},
      "display": {
        "type": "native",
        "bbox": [x0, y0, x1, y1],
        "dpi": null
      }
    }
  ]
}
```

---

## Flask Web Interface

### Module: `app.py`

**Purpose**: Simple web interface for document upload and Q&A with inline HTML template.

#### Key Functions

##### `save_current(index_dir: Path, ingest_jsonl: Path, title: str)`
Saves current dataset state to persistent storage.

```python
# Example usage
save_current(Path("rag_index"), Path("input.jsonl"), "Document Title")
```

**Parameters**:
- `index_dir`: Path to FAISS index directory
- `ingest_jsonl`: Path to ingestion JSONL file
- `title`: Document title

##### `load_current() -> Dict[str, str] | None`
Loads current dataset state from persistent storage.

```python
# Example usage
current = load_current()
# Returns: {"index_dir": "...", "ingest_jsonl": "...", "title": "..."}
```

**Returns**: Current dataset configuration or None

##### `list_recent() -> List[Dict[str, str]]`
Lists recent document uploads for switching between datasets.

```python
# Example usage
recent = list_recent()
# Returns: List of recent upload configurations
```

**Returns**: List of recent upload metadata

##### `get_default_paths() -> Dict[str, str]`
Determines the active dataset using fallback logic.

**Fallback Priority**:
1. Remembered "current" dataset (if paths exist)
2. Most recent upload
3. Demo dataset (if available)
4. Empty configuration (prompt for upload)

##### `home()`
Main page route with upload and query forms.

**Features**:
- File upload form
- Query input form
- Recent uploads list
- Search results display
- Visual citation previews

##### `upload()`
Handles PDF upload and processing.

**Process Flow**:
1. File validation (PDF only, size limits)
2. Save uploaded file
3. Run PDF ingestion (`ingest_Pdf.py`)
4. Build vector index (`build_index.py`)
5. Save as current dataset

##### `switch(key: str)`
Switches to a previous upload by key.

**Parameters**:
- `key`: Upload directory key (UUID)

##### `preview()`
Generates image previews with highlighting.

**Parameters**:
- `file`: PDF file path
- `page`: Page index
- `type`: Highlight type ("native" or "ocr")
- `dpi`: Image resolution
- `bbox`: Bounding box for highlighting

**Features**:
- Page rendering with PyMuPDF
- Visual highlighting overlay
- Configurable DPI
- PNG image output

---

## FastAPI Web Interface

### Module: `main.py`

**Purpose**: RESTful API for programmatic access to the RAG pipeline.

#### Key Functions

##### `save_current(index_dir: Path, ingest_jsonl: Path, title: str)`
Same as Flask version - saves current dataset state.

##### `load_current() -> Optional[Dict[str, str]]`
Same as Flask version - loads current dataset state.

##### `list_recent() -> List[Dict[str, str]]`
Same as Flask version - lists recent uploads.

##### `get_active_paths() -> Dict[str, str]`
Same as Flask version - determines active dataset.

##### `ensure_pdf_ext(name: str)`
Validates file extension for PDF uploads.

```python
# Example usage
ensure_pdf_ext("document.pdf")  # OK
ensure_pdf_ext("document.txt")  # Raises HTTPException
```

##### `run_ingest_and_index(pdf_path: Path, jsonl_path: Path, index_dir: Path)`
Runs the complete ingestion and indexing pipeline.

```python
# Example usage
run_ingest_and_index(
    pdf_path=Path("upload.pdf"),
    jsonl_path=Path("output.jsonl"),
    index_dir=Path("index")
)
```

**Process Flow**:
1. Run PDF ingestion subprocess
2. Run index building subprocess
3. Handle errors with HTTP exceptions

#### API Endpoints

##### `GET /health`
Health check endpoint.

**Response**:
```json
{"status": "ok"}
```

##### `GET /status`
Returns system status and recent uploads.

**Response**:
```json
{
  "active": {
    "index_dir": "...",
    "ingest_jsonl": "...",
    "title": "..."
  },
  "recent": [
    {
      "key": "abc123",
      "title": "Document.pdf",
      "index_dir": "...",
      "ingest_jsonl": "..."
    }
  ]
}
```

##### `POST /upload`
Uploads and processes a PDF document.

**Request**: Multipart form with PDF file

**Response**:
```json
{
  "message": "Upload processed",
  "key": "abc123",
  "title": "Document.pdf",
  "index_dir": "...",
  "ingest_jsonl": "..."
}
```

##### `POST /switch`
Switches to a previous upload.

**Request Body**:
```json
{"key": "abc123"}
```

**Response**:
```json
{
  "message": "Switched to Document.pdf",
  "index_dir": "...",
  "ingest_jsonl": "..."
}
```

##### `POST /qa`
Performs question answering against the active dataset.

**Request Body**:
```json
{"query": "What is the main topic?"}
```

**Response**:
```json
{
  "query": "What is the main topic?",
  "model": "llama3.1:8b",
  "answer": "LLM generated answer...",
  "results": [...]
}
```

##### `GET /preview`
Generates image previews with highlighting.

**Query Parameters**:
- `file`: PDF file path
- `page`: Page index (default: 0)
- `type`: Highlight type (default: "native")
- `dpi`: Image resolution (default: 150)
- `bbox`: Bounding box JSON (optional)

**Response**: PNG image with highlighting overlay

#### Pydantic Models

##### `QARequest`
```python
class QARequest(BaseModel):
    query: str
```

##### `QAResponse`
```python
class QAResponse(BaseModel):
    query: str
    model: str
    answer: str
    results: List[Dict[str, Any]]
```

##### `SwitchRequest`
```python
class SwitchRequest(BaseModel):
    key: str
```

---

## Configuration and Environment

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | SentenceTransformer model |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model name |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `TOP_K` | `5` | Number of search results |
| `NORMALIZE` | `True` | Enable L2 normalization |
| `TESSERACT_PATH` | `None` | Tesseract executable path |
| `OCR_LANGUAGE` | `eng` | OCR language code |

### File Structure

```
rag-pipline/
├── uploads/           # User uploaded PDFs and processed data
│   └── <uuid>/       # Per-upload directories
│       ├── *.pdf     # Original PDF
│       ├── out.jsonl # Ingestion output
│       └── index/    # Vector index
├── state/            # Application state
│   └── last_paths.json
├── rag_index/        # Demo vector index
├── json/             # Sample data
├── pdf/              # Sample PDFs
├── env/              # Virtual environment
├── app.py            # Flask web interface
├── main.py           # FastAPI web interface
├── rag_qa.py         # RAG engine
├── build_index.py    # Document indexing
├── ingest_Pdf.py     # PDF ingestion
├── search_index.py   # Vector search
├── requirements.txt  # Python dependencies
├── config.env        # Configuration file
└── README.md         # Project documentation
```

### Usage Examples

#### Command Line Usage

```bash
# 1. Ingest a PDF
python ingest_Pdf.py document.pdf --out output.jsonl --lang eng

# 2. Build vector index
python build_index.py --jsonl output.jsonl --out_dir rag_index --normalize

# 3. Search the index
python search_index.py --index_dir rag_index --ingest_jsonl output.jsonl --query "search query"

# 4. Run RAG pipeline
python rag_qa.py --index_dir rag_index --ingest_jsonl output.jsonl --query "What is the main topic?"
```

#### Web Interface Usage

```bash
# Start Flask interface
python app.py

# Start FastAPI interface
python main.py
```

#### API Usage

```python
import requests

# Upload PDF
with open("document.pdf", "rb") as f:
    response = requests.post("http://localhost:8000/upload", files={"pdf": f})

# Ask question
response = requests.post("http://localhost:8000/qa", json={"query": "What is the main topic?"})
result = response.json()
print(result["answer"])
```

### Error Handling

The system includes comprehensive error handling:

1. **File Validation**: PDF format and size checks
2. **Process Errors**: Subprocess execution failures
3. **Model Loading**: Embedding and LLM model errors
4. **API Errors**: HTTP status codes and error messages
5. **State Management**: Graceful handling of missing files

### Performance Tips

1. **Chunking**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` for optimal retrieval
2. **Batch Processing**: Use batch embedding generation for large documents
3. **Model Selection**: Choose appropriate SentenceTransformer models for your use case
4. **Index Optimization**: Use L2 normalization for better similarity matching
5. **Caching**: Models are loaded once and reused across queries

