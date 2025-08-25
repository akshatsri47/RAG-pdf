# RAG Pipeline - High-Level Design (HLD)

## 1. System Overview

The RAG (Retrieval-Augmented Generation) Pipeline is a local document question-answering system that processes PDF documents and provides intelligent responses with precise citations. The system handles both native PDFs (with embedded text) and scanned PDFs (using OCR) through a unified pipeline.

### 1.1 Key Features
- **Mixed PDF Processing**: Automatic detection and processing of native vs scanned PDFs
- **Local Processing**: All processing happens locally without external API dependencies
- **Precise Citations**: Page-level citations with visual highlighting
- **Dual Web Interfaces**: Flask-based simple UI and FastAPI-based REST API
- **Vector Search**: FAISS-based semantic search with sentence transformers
- **Local LLM Integration**: Ollama integration for local language model inference

### 1.2 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   FastAPI API   │    │   Command Line  │
│   (Flask)       │    │   (main.py)     │    │   (search_index)│
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      RAG Engine           │
                    │     (rag_qa.py)           │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    Vector Search          │
                    │   (FAISS + Embeddings)    │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    Document Index         │
                    │   (build_index.py)        │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   Document Ingestion      │
                    │   (ingest_Pdf.py)         │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │      PDF Processing       │
                    │  (PyMuPDF + Tesseract)    │
                    └───────────────────────────┘
```

## 2. Component Architecture

### 2.1 Document Processing Layer

#### 2.1.1 PDF Ingestion (`ingest_Pdf.py`)
**Purpose**: Extracts text and metadata from PDF documents with automatic native/OCR detection.

**Key Functions**:
- `page_has_native_text()`: Heuristic detection of native vs scanned pages
- `extract_native()`: Text extraction from native PDFs with block-level bounding boxes
- `extract_ocr()`: OCR processing using Tesseract with word-level bounding boxes

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
  "blocks": [{"bbox": [x0,y0,x1,y1], "text": "..."}],  // native mode
  "words": [{"bbox": [x0,y0,x1,y1], "text": "...", "conf": 89}],  // OCR mode
  "dpi": 300  // OCR mode only
}
```

### 2.2 Indexing Layer

#### 2.2.1 Document Indexing (`build_index.py`)
**Purpose**: Creates searchable vector embeddings from ingested documents.

**Key Functions**:
- `simple_chunks()`: Text chunking with overlap for context preservation
- `build_records()`: Creates chunk records with provenance and highlighting metadata
- Vector embedding generation using SentenceTransformers
- FAISS index creation and storage

**Chunking Strategy**:
- **Chunk Size**: 800 characters (configurable)
- **Overlap**: 150 characters (configurable)
- **Highlighting**: Preserves block/word indices for visual citation

**Output Files**:
- `chunks.faiss`: FAISS vector index
- `metadata.jsonl`: Chunk metadata with highlighting information

### 2.3 Search Layer

#### 2.3.1 Vector Search (`search_index.py`)
**Purpose**: Performs semantic search over indexed documents.

**Key Functions**:
- `highlight_box_for_hit()`: Computes bounding boxes for visual highlighting
- FAISS similarity search with configurable top-k results
- Score normalization and ranking

### 2.4 RAG Engine

#### 2.4.1 Question Answering (`rag_qa.py`)
**Purpose**: Orchestrates the complete RAG pipeline from query to answer.

**Key Functions**:
- `retrieve()`: Semantic retrieval of relevant document chunks
- `to_prompt()`: Constructs grounded prompts with citation instructions
- `call_ollama()`: Local LLM inference via Ollama
- `compute_bbox()`: Generates display-ready bounding boxes

**RAG Pipeline Flow**:
1. Query embedding generation
2. Vector similarity search
3. Context retrieval with highlighting metadata
4. Prompt construction with citation instructions
5. LLM inference
6. Response formatting with citations

## 3. Data Flow

### 3.1 Document Processing Flow

```
PDF Upload → Native/OCR Detection → Text Extraction → JSONL Output
     ↓              ↓                    ↓              ↓
  File Save    Page Classification   BBox Extraction   Metadata
```

### 3.2 Indexing Flow

```
JSONL Input → Text Chunking → Embedding Generation → FAISS Index
     ↓            ↓                ↓                ↓
  Page Data   Chunk Records   Vector Embeddings   Search Index
```

### 3.3 Query Processing Flow

```
User Query → Embedding → Vector Search → Context Retrieval → LLM → Response
     ↓         ↓           ↓              ↓              ↓        ↓
  Text Input  Vector    Similarity     Highlighted    Prompt   Formatted
             Query      Matching       Citations      Gen      Answer
```

## 4. Web Interface Architecture

### 4.1 Flask Interface (`app.py`)
**Purpose**: Simple web interface for document upload and Q&A.

**Key Routes**:
- `GET /`: Main interface with upload and query forms
- `POST /upload`: PDF upload and processing
- `GET /switch/<key>`: Switch between uploaded documents
- `GET /preview`: Image preview with highlighting

**Features**:
- Inline HTML template with CSS styling
- File upload with automatic processing
- Recent uploads management
- Visual citation previews

### 4.2 FastAPI Interface (`main.py`)
**Purpose**: RESTful API for programmatic access.

**Key Endpoints**:
- `GET /health`: Health check
- `GET /status`: System status and recent uploads
- `POST /upload`: PDF upload endpoint
- `POST /switch`: Switch active dataset
- `POST /qa`: Question answering endpoint
- `GET /preview`: Image preview endpoint

**Features**:
- Pydantic models for request/response validation
- CORS middleware for cross-origin requests
- Streaming responses for image previews
- Comprehensive error handling

## 5. Configuration Management

### 5.1 Environment Variables
```bash
# Embedding Model
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# LLM Configuration
OLLAMA_MODEL=llama3.1:8b
OLLAMA_URL=http://localhost:11434/api/generate

# Search Configuration
TOP_K=5
NORMALIZE=true

# OCR Configuration
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
OCR_LANGUAGE=eng
```

### 5.2 File Structure
```
rag-pipline/
├── uploads/           # User uploaded PDFs and processed data
├── state/            # Application state (current dataset)
├── rag_index/        # Vector index storage
├── json/             # Sample data and outputs
├── pdf/              # Sample PDF documents
├── env/              # Virtual environment
└── config.env        # Configuration file
```

## 6. Performance Considerations

### 6.1 Scalability
- **Chunking**: Configurable chunk size and overlap for optimal retrieval
- **Batch Processing**: Embedding generation in batches
- **Index Optimization**: FAISS normalization for better similarity matching
- **Caching**: Model loading and embedding caching

### 6.2 Memory Management
- **Streaming**: JSONL processing for large documents
- **Lazy Loading**: Models loaded on-demand
- **Garbage Collection**: Proper cleanup of large objects

### 6.3 Processing Optimization
- **Parallel Processing**: OCR and embedding generation
- **Progress Tracking**: TQDM progress bars for long operations
- **Error Handling**: Graceful failure with detailed error messages

## 7. Security Considerations

### 7.1 File Upload Security
- File type validation (PDF only)
- Secure filename handling
- File size limits (200MB max)
- Path traversal prevention

### 7.2 Data Privacy
- Local processing only
- No external API calls for sensitive data
- Temporary file cleanup
- Secure state management

## 8. Deployment Architecture

### 8.1 Development Setup
```bash
# Virtual environment
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows

# Dependencies
pip install -r requirements.txt

# External dependencies
# - Tesseract OCR
# - Ollama (for LLM inference)
```

### 8.2 Production Deployment
- **Web Server**: Gunicorn with FastAPI
- **Process Management**: Systemd or Docker
- **Monitoring**: Health check endpoints
- **Logging**: Structured logging with rotation

## 9. Future Enhancements

### 9.1 Planned Features
- **Multi-document Support**: Cross-document search and citation
- **Advanced Chunking**: Semantic chunking based on document structure
- **Caching Layer**: Redis for query result caching
- **User Management**: Multi-user support with document isolation
- **Advanced Search**: Hybrid search (keyword + semantic)

### 9.2 Performance Improvements
- **GPU Acceleration**: CUDA support for embedding generation
- **Distributed Processing**: Multi-node processing for large documents
- **Index Optimization**: Hierarchical FAISS indices
- **Streaming Processing**: Real-time document processing

### 9.3 Integration Capabilities
- **API Extensions**: Webhook support and external integrations
- **Export Formats**: Multiple output formats (JSON, CSV, PDF)
- **Plugin System**: Extensible architecture for custom processors
- **Cloud Integration**: Optional cloud storage and processing
