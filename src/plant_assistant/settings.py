"""Environment-driven settings for the plant disease assistant.

All runtime configuration is centralized here so the Streamlit UI, LangChain
agent, and MCP server use the same paths, model names, and timeout defaults.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
NOISY_LOGGERS = ("httpx", "transformers", "huggingface_hub", "sentence_transformers")


def env_path(name: str, default: str | Path) -> Path:
    """Return an absolute path from an environment variable or default value."""

    return Path(os.getenv(name, str(default))).expanduser().resolve()


def configure_logging() -> None:
    """Configure application logging and quiet common dependency loggers."""

    logging.basicConfig(
        level=os.getenv("PLANT_ASSISTANT_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# Ollama and agent execution settings.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.6:latest")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "32768"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "2m")
OLLAMA_REQUEST_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "120"))
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "180"))
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "12"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))
MAX_MEMORY_CHARS = int(os.getenv("MAX_MEMORY_CHARS", "6000"))
MAX_MESSAGE_MEMORY_CHARS = int(os.getenv("MAX_MESSAGE_MEMORY_CHARS", "700"))
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "http")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

# Local artifact paths used by the classifier, retrieval system, and uploads.
MODEL_DIR = env_path("MODEL_DIR", PROJECT_DIR / "notebooks")
CLASS_NAMES_PATH = env_path(
    "CLASS_NAMES_PATH",
    MODEL_DIR / "plant_disease_class_names.json",
)
VECTOR_DB_DIR = env_path("VECTOR_DB_DIR", PROJECT_DIR / "rag" / "chroma_db")
UPLOAD_DIR = env_path("UPLOAD_DIR", PROJECT_DIR / "data" / "uploads")

# MCP server network settings.
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
MCP_PATH = os.getenv("MCP_PATH", "/mcp")

# Default tool parameters exposed to the agent.
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "MobileNetV2")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
DEFAULT_RETRIEVAL_TOP_K = int(os.getenv("DEFAULT_RETRIEVAL_TOP_K", "4"))
DEFAULT_SHAP_MAX_EVALS = max(50, min(int(os.getenv("DEFAULT_SHAP_MAX_EVALS", "100")), 2000))
DEFAULT_SHAP_BATCH_SIZE = max(1, min(int(os.getenv("DEFAULT_SHAP_BATCH_SIZE", "8")), 100))
