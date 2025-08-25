# RAG Pipeline - Local Document Question Answering System

A comprehensive Retrieval-Augmented Generation (RAG) system for document question-answering over mixed (native + scanned) PDFs with local processing capabilities.

## 📚 Documentation

- **[High-Level Design (HLD)](ARCHITECTURE.md)** - System architecture and design overview
- **[Code Documentation](CODE_DOCUMENTATION.md)** - Detailed module and function documentation
- **[API Documentation](API_DOCUMENTATION.md)** - Complete REST API reference
- **[Configuration Guide](config.env)** - Environment variables and settings

## 🚀 Features

- **Mixed PDF Processing**: Automatic detection and processing of native vs scanned PDFs
- **Local Processing**: All processing happens locally without external API dependencies
- **Precise Citations**: Page-level citations with visual highlighting
- **Dual Web Interfaces**: Flask-based simple UI and FastAPI-based REST API
- **Vector Search**: FAISS-based semantic search with sentence transformers
- **Local LLM Integration**: Ollama integration for local language model inference
- **Visual Citations**: Image previews with highlighted text regions
- **Multi-Document Support**: Switch between different uploaded documents

## 🏗️ System Architecture

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

## 📦 Installation

### Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR** (for scanned PDFs):
   - **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`

3. **Ollama** (for local LLM):
   - Download from [ollama.ai](https://ollama.ai/)
   - Pull a model: `ollama pull llama3.1:8b`

### Setup

1. **Clone and Setup Environment**:
   ```bash
   git clone <repository-url>
   cd rag-pipline
   
   # Create virtual environment
   python -m venv env
   source env/bin/activate  # Linux/Mac
   env\Scripts\activate     # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   # Copy and edit configuration
   cp config.env.example config.env
   # Edit config.env with your settings
   ```

3. **Verify Installation**:
   ```bash
   # Test PDF ingestion
   python ingest_Pdf.py pdf/sample.pdf --out test.jsonl
   
   # Test indexing
   python build_index.py --jsonl test.jsonl --out_dir test_index
   
   # Test search
   python search_index.py --index_dir test_index --ingest_jsonl test.jsonl --query "test"
   ```

## 🎯 Quick Start

### Web Interface (Flask)

```bash
# Start Flask web interface
python app.py

# Open browser to http://localhost:5000
# Upload PDF and start asking questions!
```

### REST API (FastAPI)

```bash
# Start FastAPI server
python main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000

# API available at http://localhost:8000
```

### Command Line

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

## 🔧 Configuration

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
├── state/            # Application state (current dataset)
├── rag_index/        # Vector index storage
├── json/             # Sample data and outputs
├── pdf/              # Sample PDF documents
├── env/              # Virtual environment
├── app.py            # Flask web interface
├── main.py           # FastAPI web interface
├── rag_qa.py         # RAG engine
├── build_index.py    # Document indexing
├── ingest_Pdf.py     # PDF ingestion
├── search_index.py   # Vector search
├── requirements.txt  # Python dependencies
├── config.env        # Configuration file
├── ARCHITECTURE.md   # High-level design
├── CODE_DOCUMENTATION.md # Code documentation
├── API_DOCUMENTATION.md  # API reference
└── README.md         # This file
```

## 📖 Usage Examples

### Python API Client

```python
import requests

# Upload PDF
with open("document.pdf", "rb") as f:
    response = requests.post("http://localhost:8000/upload", files={"pdf": f})
    upload_data = response.json()
    print(f"Uploaded: {upload_data['title']}")

# Ask questions
response = requests.post("http://localhost:8000/qa", 
                        json={"query": "What is the main topic?"})
result = response.json()
print(f"Answer: {result['answer']}")

# Get citations
for citation in result['results']:
    print(f"Page {citation['citation']['page_label']}: {citation['text'][:100]}...")
```

### Batch Processing

```python
import requests
from pathlib import Path

def process_documents(pdf_dir):
    for pdf_file in Path(pdf_dir).glob("*.pdf"):
        # Upload
        with open(pdf_file, "rb") as f:
            response = requests.post("http://localhost:8000/upload", files={"pdf": f})
            if response.status_code == 200:
                print(f"✓ Processed {pdf_file.name}")
            else:
                print(f"✗ Failed {pdf_file.name}")

# Process all PDFs in directory
process_documents("documents/")
```

## 🔍 How It Works

### 1. Document Processing
- **Native PDFs**: Direct text extraction with PyMuPDF
- **Scanned PDFs**: OCR processing with Tesseract
- **Mixed Documents**: Automatic detection per page
- **Output**: JSONL with text, metadata, and bounding boxes

### 2. Indexing
- **Chunking**: Intelligent text chunking with overlap
- **Embeddings**: SentenceTransformer vector generation
- **Storage**: FAISS index for fast similarity search
- **Metadata**: Preserves highlighting information

### 3. Question Answering
- **Query Embedding**: Convert question to vector
- **Semantic Search**: Find most relevant document chunks
- **Context Retrieval**: Extract highlighted text regions
- **LLM Generation**: Local inference with Ollama
- **Citation Generation**: Page-level citations with evidence

### 4. Visual Citations
- **Bounding Boxes**: Precise text region highlighting
- **Image Rendering**: PDF page to image conversion
- **Overlay Generation**: Visual highlighting on previews
- **Coordinate Mapping**: PDF points to pixel conversion

## 🚀 Performance

### Optimization Tips

1. **Chunking**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` for optimal retrieval
2. **Batch Processing**: Use batch embedding generation for large documents
3. **Model Selection**: Choose appropriate SentenceTransformer models
4. **Index Optimization**: Use L2 normalization for better similarity matching
5. **Caching**: Models are loaded once and reused across queries

### Benchmarks

- **PDF Processing**: ~1-2 seconds per page (native), ~5-10 seconds per page (OCR)
- **Indexing**: ~1000 chunks/second with GPU acceleration
- **Query Response**: ~1-3 seconds for typical questions
- **Memory Usage**: ~2-4GB for typical documents

## 🔧 Troubleshooting

### Common Issues

1. **Tesseract not found**:
   ```bash
   # Windows: Set path in config.env
   TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
   
   # Linux/macOS: Install via package manager
   sudo apt-get install tesseract-ocr  # Ubuntu
   brew install tesseract              # macOS
   ```

2. **Ollama not running**:
   ```bash
   # Start Ollama service
   ollama serve
   
   # Pull required model
   ollama pull llama3.1:8b
   ```

3. **Memory issues**:
   ```bash
   # Reduce chunk size in build_index.py
   CHUNK_SIZE = 400  # Default: 800
   
   # Use smaller embedding model
   EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```

4. **Slow processing**:
   ```bash
   # Enable GPU acceleration (if available)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Use smaller models for faster inference
   OLLAMA_MODEL=llama3.1:1b
   ```

### Debug Mode

```bash
# Enable debug logging
export DEBUG=True
python main.py

# Check logs for detailed error information
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
isort .

# Lint code
flake8 .
mypy .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyMuPDF**: PDF processing and text extraction
- **Tesseract**: OCR capabilities
- **FAISS**: Vector similarity search
- **SentenceTransformers**: Text embeddings
- **Ollama**: Local LLM inference
- **FastAPI**: Modern web API framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: See the [documentation files](#-documentation) above

---

**Made with ❤️ for local, privacy-preserving document AI**


