# RAG Pipeline - API Documentation

## Overview

The RAG Pipeline provides a RESTful API for document question-answering with local processing capabilities. The API supports PDF upload, document indexing, semantic search, and question answering with precise citations.

**Base URL**: `http://localhost:8000`  
**Content Type**: `application/json` (except for file uploads)  
**Authentication**: None (local processing)

## Quick Start

### 1. Start the API Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
python main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Basic Usage

```python
import requests

# Upload a PDF
with open("document.pdf", "rb") as f:
    response = requests.post("http://localhost:8000/upload", files={"pdf": f})
    upload_data = response.json()
    print(f"Uploaded: {upload_data['title']}")

# Ask a question
response = requests.post("http://localhost:8000/qa", json={"query": "What is the main topic?"})
result = response.json()
print(f"Answer: {result['answer']}")
```

## Endpoints

### Health Check

#### `GET /health`

Check if the API server is running.

**Response**:
```json
{
  "status": "ok"
}
```

**Status Codes**:
- `200 OK`: Server is healthy

**Example**:
```bash
curl http://localhost:8000/health
```

### System Status

#### `GET /status`

Get current system status including active dataset and recent uploads.

**Response**:
```json
{
  "active": {
    "index_dir": "/path/to/rag_index",
    "ingest_jsonl": "/path/to/input.jsonl",
    "title": "Document.pdf"
  },
  "recent": [
    {
      "key": "abc12345",
      "title": "Document.pdf",
      "index_dir": "/path/to/uploads/abc12345/index",
      "ingest_jsonl": "/path/to/uploads/abc12345/out.jsonl"
    }
  ]
}
```

**Status Codes**:
- `200 OK`: Status retrieved successfully

**Example**:
```bash
curl http://localhost:8000/status
```

### Document Upload

#### `POST /upload`

Upload and process a PDF document. The document will be automatically ingested and indexed for question answering.

**Content Type**: `multipart/form-data`

**Form Fields**:
- `pdf` (file, required): PDF file to upload (max 200MB)

**Response**:
```json
{
  "message": "Upload processed",
  "key": "abc12345",
  "title": "Document.pdf",
  "index_dir": "/path/to/uploads/abc12345/index",
  "ingest_jsonl": "/path/to/uploads/abc12345/out.jsonl"
}
```

**Status Codes**:
- `200 OK`: Upload processed successfully
- `400 Bad Request`: Invalid file type or missing file
- `500 Internal Server Error`: Processing failed

**Processing Steps**:
1. File validation (PDF format, size limits)
2. PDF ingestion (text extraction + OCR if needed)
3. Vector index building
4. Dataset activation

**Example**:
```bash
curl -X POST http://localhost:8000/upload \
  -F "pdf=@document.pdf"
```

**Python Example**:
```python
import requests

with open("document.pdf", "rb") as f:
    files = {"pdf": f}
    response = requests.post("http://localhost:8000/upload", files=files)
    
if response.status_code == 200:
    data = response.json()
    print(f"Uploaded: {data['title']}")
    print(f"Key: {data['key']}")
else:
    print(f"Upload failed: {response.text}")
```

### Dataset Switching

#### `POST /switch`

Switch to a previously uploaded document for question answering.

**Request Body**:
```json
{
  "key": "abc12345"
}
```

**Response**:
```json
{
  "message": "Switched to Document.pdf",
  "index_dir": "/path/to/uploads/abc12345/index",
  "ingest_jsonl": "/path/to/uploads/abc12345/out.jsonl"
}
```

**Status Codes**:
- `200 OK`: Dataset switched successfully
- `404 Not Found`: Upload key not found or incomplete

**Example**:
```bash
curl -X POST http://localhost:8000/switch \
  -H "Content-Type: application/json" \
  -d '{"key": "abc12345"}'
```

**Python Example**:
```python
import requests

response = requests.post("http://localhost:8000/switch", 
                        json={"key": "abc12345"})

if response.status_code == 200:
    data = response.json()
    print(f"Switched to: {data['message']}")
else:
    print(f"Switch failed: {response.text}")
```

### Question Answering

#### `POST /qa`

Ask a question against the currently active document dataset.

**Request Body**:
```json
{
  "query": "What is the main topic of this document?"
}
```

**Response**:
```json
{
  "query": "What is the main topic of this document?",
  "model": "llama3.1:8b",
  "answer": "Based on the provided context, the main topic of this document is artificial intelligence and machine learning applications. The document discusses various AI techniques including neural networks, deep learning, and their practical implementations in real-world scenarios. [p:1] [p:3]\n\nEvidence:\n- \"The field of artificial intelligence has seen remarkable progress in recent years\" (p:1)\n- \"Machine learning algorithms are being applied across diverse domains\" (p:3)",
  "results": [
    {
      "rank": 1,
      "score": 0.8923,
      "text": "The field of artificial intelligence has seen remarkable progress in recent years, with breakthroughs in machine learning, natural language processing, and computer vision.",
      "citation": {
        "file": "/path/to/document.pdf",
        "page_index": 0,
        "page_label": "1",
        "mode": "native"
      },
      "highlight": {
        "type": "native",
        "block_indices": [0, 1, 2]
      },
      "display": {
        "type": "native",
        "bbox": [72.0, 72.0, 540.0, 120.0],
        "dpi": null
      }
    }
  ]
}
```

**Status Codes**:
- `200 OK`: Question answered successfully
- `400 Bad Request`: No active dataset or missing query
- `500 Internal Server Error`: Processing failed

**Response Fields**:
- `query`: Original user question
- `model`: LLM model used for generation
- `answer`: Generated answer with citations
- `results`: Retrieved document chunks with metadata

**Example**:
```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?"}'
```

**Python Example**:
```python
import requests

response = requests.post("http://localhost:8000/qa", 
                        json={"query": "What is the main topic?"})

if response.status_code == 200:
    result = response.json()
    print(f"Answer: {result['answer']}")
    
    # Print citations
    for i, citation in enumerate(result['results'], 1):
        print(f"Citation {i}: Page {citation['citation']['page_label']}")
else:
    print(f"QA failed: {response.text}")
```

### Image Preview

#### `GET /preview`

Generate a preview image of a PDF page with optional highlighting.

**Query Parameters**:
- `file` (string, required): Absolute path to the PDF file
- `page` (integer, optional): Page index (0-based, default: 0)
- `type` (string, optional): Highlight type - "native" or "ocr" (default: "native")
- `dpi` (integer, optional): Image resolution (100-600, default: 150)
- `bbox` (string, optional): JSON array of bounding box coordinates [x0, y0, x1, y1]

**Response**: PNG image with highlighting overlay

**Status Codes**:
- `200 OK`: Image generated successfully
- `400 Bad Request`: Invalid parameters or page out of range
- `404 Not Found`: PDF file not found

**Example**:
```bash
# Basic preview
curl "http://localhost:8000/preview?file=/path/to/document.pdf&page=0" \
  -o preview.png

# With highlighting
curl "http://localhost:8000/preview?file=/path/to/document.pdf&page=0&bbox=[100,100,200,150]" \
  -o highlighted.png
```

**Python Example**:
```python
import requests

# Get basic preview
params = {
    "file": "/path/to/document.pdf",
    "page": 0,
    "dpi": 200
}
response = requests.get("http://localhost:8000/preview", params=params)

if response.status_code == 200:
    with open("preview.png", "wb") as f:
        f.write(response.content)
    print("Preview saved as preview.png")

# Get preview with highlighting
params["bbox"] = "[100, 100, 200, 150]"
response = requests.get("http://localhost:8000/preview", params=params)

if response.status_code == 200:
    with open("highlighted.png", "wb") as f:
        f.write(response.content)
    print("Highlighted preview saved")
```

## Data Models

### Request Models

#### QARequest
```python
class QARequest(BaseModel):
    query: str
```

#### SwitchRequest
```python
class SwitchRequest(BaseModel):
    key: str
```

### Response Models

#### QAResponse
```python
class QAResponse(BaseModel):
    query: str
    model: str
    answer: str
    results: List[Dict[str, Any]]
```

#### Citation
```python
class Citation(BaseModel):
    file: str
    page_index: int
    page_label: Optional[str]
    mode: str  # "native" or "ocr"
```

#### DisplayInfo
```python
class DisplayInfo(BaseModel):
    type: str  # "native" or "ocr"
    bbox: Optional[List[float]]  # [x0, y0, x1, y1]
    dpi: Optional[int]
```

## Error Handling

### Error Response Format

All error responses follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

| Status Code | Description | Common Causes |
|-------------|-------------|---------------|
| `400 Bad Request` | Invalid request parameters | Missing required fields, invalid file type |
| `404 Not Found` | Resource not found | File not found, upload key doesn't exist |
| `500 Internal Server Error` | Server processing error | OCR failure, indexing error, LLM error |

### Error Examples

#### File Upload Error
```json
{
  "detail": "Only .pdf files are allowed"
}
```

#### Missing Dataset Error
```json
{
  "detail": "No active dataset. Upload a PDF first."
}
```

#### Processing Error
```json
{
  "detail": "Ingestion failed: Tesseract not found"
}
```

## Configuration

### Environment Variables

The API uses the following environment variables (can be set in `config.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `OLLAMA_MODEL` | `llama3.1:8b` | LLM model name |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API URL |
| `TOP_K` | `5` | Number of search results |
| `NORMALIZE` | `True` | Enable L2 normalization |
| `TESSERACT_PATH` | `None` | Tesseract executable path |
| `OCR_LANGUAGE` | `eng` | OCR language |

### File Size Limits

- **Maximum PDF size**: 200MB
- **Supported formats**: PDF only
- **Processing timeout**: 120 seconds per request

## Usage Examples

### Complete Workflow

```python
import requests
import time

# 1. Check server health
response = requests.get("http://localhost:8000/health")
print(f"Server status: {response.json()}")

# 2. Upload a PDF
with open("document.pdf", "rb") as f:
    response = requests.post("http://localhost:8000/upload", files={"pdf": f})
    if response.status_code == 200:
        upload_data = response.json()
        print(f"Uploaded: {upload_data['title']}")
    else:
        print(f"Upload failed: {response.text}")
        exit(1)

# 3. Wait for processing (optional)
time.sleep(2)

# 4. Ask questions
questions = [
    "What is the main topic?",
    "What are the key findings?",
    "Who are the authors?"
]

for question in questions:
    response = requests.post("http://localhost:8000/qa", 
                           json={"query": question})
    if response.status_code == 200:
        result = response.json()
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
    else:
        print(f"QA failed for '{question}': {response.text}")

# 5. Get preview images
response = requests.get("http://localhost:8000/status")
if response.status_code == 200:
    status = response.json()
    if status['active']['index_dir']:
        # Get preview of first page
        preview_params = {
            "file": status['active']['ingest_jsonl'].replace('out.jsonl', 'document.pdf'),
            "page": 0,
            "dpi": 200
        }
        response = requests.get("http://localhost:8000/preview", 
                              params=preview_params)
        if response.status_code == 200:
            with open("preview.png", "wb") as f:
                f.write(response.content)
            print("Preview saved as preview.png")
```

### Batch Processing

```python
import requests
import os
from pathlib import Path

def process_document(pdf_path):
    """Process a single PDF document and return results."""
    
    # Upload
    with open(pdf_path, "rb") as f:
        response = requests.post("http://localhost:8000/upload", files={"pdf": f})
        if response.status_code != 200:
            return None
        upload_data = response.json()
    
    # Ask questions
    questions = ["What is this document about?", "What are the main points?"]
    results = []
    
    for question in questions:
        response = requests.post("http://localhost:8000/qa", 
                               json={"query": question})
        if response.status_code == 200:
            results.append(response.json())
    
    return {
        "file": pdf_path.name,
        "upload": upload_data,
        "qa_results": results
    }

# Process multiple documents
pdf_dir = Path("documents")
results = []

for pdf_file in pdf_dir.glob("*.pdf"):
    print(f"Processing {pdf_file.name}...")
    result = process_document(pdf_file)
    if result:
        results.append(result)
        print(f"✓ Completed {pdf_file.name}")
    else:
        print(f"✗ Failed {pdf_file.name}")

# Save results
import json
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Integration with External Systems

```python
import requests
import json
from typing import Dict, Any

class RAGClient:
    """Client for interacting with the RAG Pipeline API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def upload_document(self, pdf_path: str) -> Dict[str, Any]:
        """Upload and process a PDF document."""
        with open(pdf_path, "rb") as f:
            response = requests.post(f"{self.base_url}/upload", files={"pdf": f})
            response.raise_for_status()
            return response.json()
    
    def ask_question(self, query: str) -> Dict[str, Any]:
        """Ask a question about the current document."""
        response = requests.post(f"{self.base_url}/qa", 
                               json={"query": query})
        response.raise_for_status()
        return response.json()
    
    def switch_document(self, key: str) -> Dict[str, Any]:
        """Switch to a different uploaded document."""
        response = requests.post(f"{self.base_url}/switch", 
                               json={"key": key})
        response.raise_for_status()
        return response.json()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        response = requests.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()

# Usage
client = RAGClient()

# Upload document
result = client.upload_document("document.pdf")
print(f"Uploaded: {result['title']}")

# Ask questions
questions = ["What is the main topic?", "What are the conclusions?"]
for question in questions:
    answer = client.ask_question(question)
    print(f"Q: {question}")
    print(f"A: {answer['answer']}\n")
```

## Performance Considerations

### Optimization Tips

1. **Batch Processing**: Upload multiple documents and switch between them
2. **Caching**: The API caches models and embeddings for better performance
3. **Chunking**: Adjust chunk size in configuration for optimal retrieval
4. **Model Selection**: Choose appropriate embedding models for your use case

### Rate Limiting

Currently, the API does not implement rate limiting. For production use, consider:

- Implementing request rate limiting
- Adding authentication/authorization
- Using a reverse proxy (nginx) for load balancing

### Monitoring

Monitor the following metrics:

- Response times for upload and QA endpoints
- Memory usage during document processing
- Disk space for uploaded documents and indices
- LLM inference latency

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Install Tesseract OCR and set `TESSERACT_PATH`
2. **Ollama not running**: Start Ollama service and ensure model is downloaded
3. **Memory issues**: Reduce chunk size or use smaller embedding models
4. **Slow processing**: Use GPU acceleration for embedding generation

### Debug Mode

Enable debug logging by setting environment variables:

```bash
export PYTHONPATH=.
export DEBUG=True
python main.py
```

### Health Checks

Regular health checks:

```python
import requests

def check_api_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Monitor health
if not check_api_health():
    print("API is not responding")
```

