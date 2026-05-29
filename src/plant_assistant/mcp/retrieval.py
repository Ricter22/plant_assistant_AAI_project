"""Local vector database retrieval for plant disease resources."""

from __future__ import annotations

import logging
from typing import Any

from plant_assistant import settings
from plant_assistant.mcp.constants import VECTOR_DB_COLLECTION_NAME, VECTOR_DB_EMBEDDING_MODEL


logger = logging.getLogger(__name__)

_vector_db: Any | None = None


def load_vector_db() -> Any:
    """Load and cache the local Chroma vector database used for retrieval."""

    global _vector_db

    if _vector_db is not None:
        return _vector_db

    if not settings.VECTOR_DB_DIR.exists():
        raise FileNotFoundError(
            f"Plant disease vector DB not found: {settings.VECTOR_DB_DIR}. "
            "Build it with AAI_project/rag/build_vector_db.ipynb first."
        )

    try:
        # Retrieval imports are kept local so classification-only deployments do
        # not pay import cost until the retrieval tool is used.
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError(
            "Plant disease retrieval dependencies are missing. Install them with: "
            "pip install chromadb sentence-transformers langchain-chroma langchain-huggingface"
        ) from exc

    logger.info(
        "Loading vector DB collection=%s path=%s embedding=%s",
        VECTOR_DB_COLLECTION_NAME,
        settings.VECTOR_DB_DIR,
        VECTOR_DB_EMBEDDING_MODEL,
    )
    embedding = HuggingFaceEmbeddings(model_name=VECTOR_DB_EMBEDDING_MODEL)
    _vector_db = Chroma(
        collection_name=VECTOR_DB_COLLECTION_NAME,
        persist_directory=str(settings.VECTOR_DB_DIR),
        embedding_function=embedding,
    )
    return _vector_db


def build_retrieval_result(query: str, top_k: int) -> dict[str, Any]:
    """Run vector search and serialize document metadata for the agent."""

    vector_db = load_vector_db()
    safe_top_k = max(1, min(int(top_k), 10))
    logger.info("Running retrieval top_k=%s query=%r", safe_top_k, query)
    results = [
        {
            "rank": rank,
            "score": float(score),
            "content": document.page_content,
            "title": str(document.metadata.get("title", "")),
            "labels": str(document.metadata.get("labels", "")),
            "source_url": str(document.metadata.get("source_url", "")),
            "final_url": str(document.metadata.get("final_url", "")),
            "resource_id": str(document.metadata.get("resource_id", "")),
            "chunk_index": int(document.metadata.get("chunk_index", 0)),
        }
        for rank, (document, score) in enumerate(
            vector_db.similarity_search_with_score(query, k=safe_top_k),
            start=1,
        )
    ]
    result = {
        "query": query,
        "result_count": len(results),
        "results": results,
    }
    logger.info("Retrieval completed result_count=%s", result["result_count"])
    return result
