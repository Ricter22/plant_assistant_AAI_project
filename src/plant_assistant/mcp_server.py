"""Executable compatibility wrapper for the plant disease MCP server.

The implementation is split across ``plant_assistant.mcp`` modules, while this
module preserves existing imports and ``python -m plant_assistant.mcp_server``.
"""

from __future__ import annotations

import logging
import os

from plant_assistant import settings
from plant_assistant.mcp.prediction import build_model_comparison_result, build_prediction_result
from plant_assistant.mcp.retrieval import build_retrieval_result, load_vector_db
from plant_assistant.mcp.tools import (
    compare_plant_disease_models,
    mcp,
    predict_plant_disease,
    retrieve_plant_disease_info,
)


logger = logging.getLogger(__name__)


__all__ = [
    "build_model_comparison_result",
    "build_prediction_result",
    "build_retrieval_result",
    "compare_plant_disease_models",
    "mcp",
    "predict_plant_disease",
    "retrieve_plant_disease_info",
    "should_preload_vector_db",
]


def should_preload_vector_db() -> bool:
    """Read the opt-in flag that warms the vector DB before serving requests."""

    value = os.getenv("MCP_PRELOAD_VECTOR_DB", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    # Module execution starts the MCP server. HTTP is the default for local app
    # use, while stdio is available for subprocess-based agent connections.
    settings.configure_logging()
    if should_preload_vector_db():
        logger.info("Preloading vector DB before serving MCP requests")
        load_vector_db()
        logger.info("Vector DB preload completed")

    transport = os.getenv("MCP_TRANSPORT", "http")
    logger.info(
        "Starting MCP server transport=%s host=%s port=%s path=%s",
        transport,
        settings.MCP_HOST,
        settings.MCP_PORT,
        settings.MCP_PATH,
    )
    if transport == "stdio":
        mcp.run(transport=transport)
    else:
        mcp.run(
            transport=transport,
            host=settings.MCP_HOST,
            port=settings.MCP_PORT,
            path=settings.MCP_PATH,
        )
