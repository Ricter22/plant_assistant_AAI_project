"""Constants and output schemas for plant disease MCP tools."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from plant_assistant import settings


# Directory anchors are used when resolving relative image paths supplied by the
# chat UI or by a user prompt.
SERVER_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SERVER_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent

IMG_SIZE = (224, 224)
SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

# The public model names below are part of the MCP tool interface.
MODEL_PATHS = {
    "MobileNetV2": settings.MODEL_DIR / "plant_disease_mobilenetv2.keras",
    "Custom CNN": settings.MODEL_DIR / "plant_disease_custom_cnn.keras",
}
MODEL_ALIASES = {
    "mobilenet": "MobileNetV2",
    "mobilenetv2": "MobileNetV2",
    "plant_disease_mobilenetv2": "MobileNetV2",
    "custom": "Custom CNN",
    "custom_cnn": "Custom CNN",
    "plant_disease_custom_cnn": "Custom CNN",
}
VECTOR_DB_COLLECTION_NAME = "plant_disease_resources"
VECTOR_DB_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ModelName = Literal[
    "MobileNetV2",
    "Custom CNN",
    "mobilenet",
    "mobilenetv2",
    "plant_disease_mobilenetv2",
    "custom",
    "custom_cnn",
    "plant_disease_custom_cnn",
]

# Tool output schemas are intentionally explicit so the agent receives stable
# structured data and reviewers can see the exact response contract.
PREDICTION_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "model",
        "image",
        "top_class_index",
        "top_class",
        "top_confidence",
        "predictions",
        "shap",
    ],
    "properties": {
        "model": {"type": "string", "enum": ["MobileNetV2", "Custom CNN"]},
        "image": {"type": "string"},
        "top_class_index": {"type": "integer", "minimum": 0},
        "top_class": {"type": "string"},
        "top_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "predictions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["rank", "class", "confidence"],
                "properties": {
                    "rank": {"type": "integer", "minimum": 1},
                    "class": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
        },
        "shap": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["included"],
                    "properties": {"included": {"const": False}},
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "included",
                        "mime_type",
                        "explained_class",
                        "max_evals",
                        "batch_size",
                    ],
                    "properties": {
                        "included": {"const": True},
                        "mime_type": {"const": "image/png"},
                        "explained_class": {"type": "string"},
                        "max_evals": {"type": "integer", "minimum": 50},
                        "batch_size": {"type": "integer", "minimum": 1},
                    },
                },
            ]
        },
    },
}
RETRIEVAL_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["query", "result_count", "results"],
    "properties": {
        "query": {"type": "string"},
        "result_count": {"type": "integer", "minimum": 0},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "rank",
                    "score",
                    "content",
                    "title",
                    "labels",
                    "source_url",
                    "final_url",
                    "resource_id",
                    "chunk_index",
                ],
                "properties": {
                    "rank": {"type": "integer", "minimum": 1},
                    "score": {"type": "number"},
                    "content": {"type": "string"},
                    "title": {"type": "string"},
                    "labels": {"type": "string"},
                    "source_url": {"type": "string"},
                    "final_url": {"type": "string"},
                    "resource_id": {"type": "string"},
                    "chunk_index": {"type": "integer", "minimum": 0},
                },
            },
        },
    },
}
COMPARISON_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "image",
        "top_k",
        "agreement",
        "models",
        "confidence_gap",
        "summary",
    ],
    "properties": {
        "image": {"type": "string"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
        "agreement": {"type": "boolean"},
        "models": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "model",
                    "top_class_index",
                    "top_class",
                    "top_confidence",
                    "predictions",
                ],
                "properties": {
                    "model": {"type": "string", "enum": ["MobileNetV2", "Custom CNN"]},
                    "top_class_index": {"type": "integer", "minimum": 0},
                    "top_class": {"type": "string"},
                    "top_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "predictions": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["rank", "class", "confidence"],
                            "properties": {
                                "rank": {"type": "integer", "minimum": 1},
                                "class": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            },
                        },
                    },
                },
            },
        },
        "confidence_gap": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "summary": {"type": "string", "enum": ["models_agree", "models_disagree"]},
    },
}
