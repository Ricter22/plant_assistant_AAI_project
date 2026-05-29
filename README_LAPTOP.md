# Laptop Run Guide

This package starts the MCP server and Streamlit directly on your laptop.

## 1. Install Ollama

Install Ollama for Windows from:

```text
https://ollama.com/download
```

Then pull the configured multimodal model:

```powershell
ollama pull qwen3.6:latest
```

Keep Ollama running in the background.

## 2. Install Python

Use Python 3.11 or 3.12. TensorFlow officially supports Python 3.9-3.12.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
```

## 3. Run

```powershell
python scripts\run_local.py
```

Useful runtime overrides:

```powershell
$env:OLLAMA_NUM_CTX="32768"
$env:OLLAMA_NUM_PREDICT="1024"
$env:OLLAMA_KEEP_ALIVE="2m"
$env:OLLAMA_REQUEST_TIMEOUT_SECONDS="120"
$env:AGENT_TIMEOUT_SECONDS="180"
$env:AGENT_RECURSION_LIMIT="12"
$env:MAX_HISTORY_TURNS="6"
$env:MAX_MEMORY_CHARS="6000"
$env:MAX_MESSAGE_MEMORY_CHARS="700"
$env:DEFAULT_SHAP_MAX_EVALS="100"
$env:DEFAULT_SHAP_BATCH_SIZE="8"
python scripts\run_local.py
```

Open:

```text
http://localhost:8501
```

## 4. Smoke Test

```powershell
$env:PYTHONPATH="src"
python scripts\smoke_check.py --skip-retrieval --prediction-image samples\AppleCedarRust1.JPG
python scripts\smoke_check.py
```

The first retrieval run may download the embedding model `sentence-transformers/all-MiniLM-L6-v2`.

## Notes About ChromaDB

The ChromaDB vector store is included in `rag/chroma_db`. It should work on Windows because it is a persisted SQLite/HNSW store, but you must copy the entire directory, not only `chroma.sqlite3`.

If Chroma fails after dependency upgrades, recreate the vector database from the original resources or use the dependency versions in this package first.
