# Plant Disease Assistant

This project runs a local plant disease assistant from Python. The runtime app is separate from the experimental notebooks:

- Streamlit provides the user interface.
- A LangChain agent talks to Ollama and the MCP tools.
- The MCP server exposes plant disease prediction and local RAG retrieval.
- Ollama runs locally and serves the configured model.
- Trained `.keras` models and the Chroma vector database are loaded as local assets.

## Setup

This project requires a gpu as the qwen model selected needs around 25GB of VRAM. For the project development I used a B200, but smaller machines are suitable as well.

Install Ollama and pull the configured multimodal model:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen3.6:latest
```

Create a Python environment and install the local runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
```

## Run

```bash
python scripts/run_local.py
```

Open the UI at:

```text
http://localhost:8501
```

## Required Local Assets

These files must exist before starting the app:

```text
notebooks/plant_disease_mobilenetv2.keras
notebooks/plant_disease_custom_cnn.keras
notebooks/plant_disease_class_names.json
rag/chroma_db/chroma.sqlite3
```

The existing notebooks document how the models and vector database were created. They are not part of the runtime path.

## Repository Contents

- `data/` stores local runtime data created or used by the app. The `data/uploads/` folder is where uploaded plant images are placed during local use.

- `datasets/` contains project datasets and supporting source material. `datasets/plant-diseases/` includes the compact reviewer test images, while `datasets/resources/` contains normalized Markdown plant disease references, an index, download helper files, and useful links for the RAG pipeline.

- `notebooks/` contains the experimental notebooks used to train and test the plant disease models. It also stores the trained `.keras` models and class-name mapping used by the runtime app.

- `rag/` contains retrieval-augmented generation assets. `rag/chroma_db/` is the persisted Chroma vector database used for local disease-management retrieval.

- `scripts/` contains local helper entrypoints. `run_local.py` starts the MCP server and Streamlit app, and `smoke_check.py` verifies that required assets, retrieval, and prediction paths work.

- `src/` contains the runtime Python package. `src/plant_assistant/` holds app settings, health checks, the MCP server, agent logic, MCP tools/resources/retrieval code, and the Streamlit UI.

The repository intentionally includes only the compact reviewer test images in `datasets/plant-diseases/test/test`. The full augmented training and validation dataset is excluded from git because it is large. The downloaded raw RAG source files are also excluded; the normalized Markdown resources, resource index, and persisted Chroma vector database are included so the app and smoke checks can run locally.

## Configuration

Useful environment variables:

```text
OLLAMA_MODEL=qwen3.6:latest
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_NUM_CTX=32768
OLLAMA_NUM_PREDICT=1024
OLLAMA_KEEP_ALIVE=2m
OLLAMA_REQUEST_TIMEOUT_SECONDS=120
AGENT_TIMEOUT_SECONDS=180
AGENT_RECURSION_LIMIT=12
MAX_HISTORY_TURNS=6
MAX_MEMORY_CHARS=6000
MAX_MESSAGE_MEMORY_CHARS=700
MCP_SERVER_URL=http://localhost:8000/mcp
MODEL_DIR=notebooks
CLASS_NAMES_PATH=notebooks/plant_disease_class_names.json
VECTOR_DB_DIR=rag/chroma_db
UPLOAD_DIR=data/uploads
DEFAULT_SHAP_MAX_EVALS=100
DEFAULT_SHAP_BATCH_SIZE=8
```
