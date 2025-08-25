# Local RAG System for Document QA

A comprehensive Retrieval-Augmented Generation (RAG) system for document question-answering over mixed (native + scanned) PDFs.

## Features

- **Mixed PDF Processing**: Handles both native PDFs (text extraction) and scanned PDFs (OCR)
- **Intelligent Chunking**: Semantic document chunking for optimal retrieval
- **Local Embeddings**: Uses local open-source embedding models
- **Vector Storage**: Local ChromaDB for vector storage
- **Local LLM Integration**: Works with Ollama for local LLM inference
- **Precise Citations**: Includes page numbers and highlights cited text
- **Web UI**: Local web interface for PDF upload and chat

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Tesseract OCR** (for scanned PDFs):
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

3. **Install Ollama** (for local LLM):
   - Download from https://ollama.ai/
   - Pull a model: `ollama pull llama2`

4. **Environment Setup**:
   ```bash
   cp .env.example .env
   # Edit .env with your configurations
   ```

## Usage

1. **Start the Web UI**:
   ```bash
   python main.py
   ```

2. **Upload PDFs** through the web interface

3. **Ask questions** and get answers with citations

## Project Structure

```
rag-pipline/
├── src/
│   ├── pdf_processor.py      # PDF processing and OCR
│   ├── text_chunker.py       # Intelligent text chunking
│   ├── embeddings.py         # Embedding generation
│   ├── vector_store.py       # Vector database operations
│   ├── rag_engine.py         # Main RAG logic
│   └── citation_handler.py   # Citation management
├── web/
│   ├── app.py               # FastAPI web application
│   ├── templates/           # HTML templates
│   └── static/              # CSS/JS files
├── data/
│   ├── pdfs/               # Uploaded PDFs
│   ├── chunks/             # Processed text chunks
│   └── vectors/            # Vector database
├── main.py                 # Application entry point
└── requirements.txt        # Dependencies
```

## Configuration

Edit `.env` file to configure:
- Tesseract path
- Ollama model name
- Embedding model
- Chunk size and overlap
- Vector database settings


