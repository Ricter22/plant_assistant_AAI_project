from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from mcp.types import ImageContent, TextContent
import numpy as np
from pydantic import Field


SERVER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SERVER_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent

IMG_SIZE = (224, 224)
SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

MODEL_PATHS = {
    "MobileNetV2": SERVER_DIR / "plant_disease_mobilenetv2.keras",
    "Custom CNN": SERVER_DIR / "plant_disease_custom_cnn.keras",
}
MODEL_ALIASES = {
    "mobilenet": "MobileNetV2",
    "mobilenetv2": "MobileNetV2",
    "plant_disease_mobilenetv2": "MobileNetV2",
    "custom": "Custom CNN",
    "custom_cnn": "Custom CNN",
    "plant_disease_custom_cnn": "Custom CNN",
}
CLASS_NAMES_PATH = SERVER_DIR / "plant_disease_class_names.json"
VECTOR_DB_DIR = PROJECT_DIR / "rag" / "chroma_db"
VECTOR_DB_COLLECTION_NAME = "plant_disease_resources"
VECTOR_DB_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
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

_tf: Any | None = None
_models: dict[str, Any] = {}
_class_names: list[str] | None = None
_vector_db: Any | None = None

mcp = FastMCP(
    "PlantDiseaseAssistant",
    instructions=(
        "Use predict_plant_disease to classify local leaf images with the "
        "already-trained plant disease CNN models. Set include_shap=true only "
        "when the user asks for an explanation image or SHAP visualization. "
        "Use retrieve_plant_disease_info to look up symptoms, prevention, "
        "care, and management information from the local plant disease vector DB."
    ),
    strict_input_validation=True,
)


def resolve_image_path(image_path: str | Path) -> Path:
    raw_path = Path(image_path).expanduser()
    candidates = [raw_path]

    if not raw_path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / raw_path,
                SERVER_DIR / raw_path,
                PROJECT_DIR / raw_path,
                PROJECT_DIR / "datasets" / raw_path,
                WORKSPACE_DIR / raw_path,
            ]
        )

    for candidate in candidates:
        path = candidate.resolve()
        if path.exists():
            break
    else:
        checked = "\n".join(f"- {candidate.resolve()}" for candidate in candidates)
        raise FileNotFoundError(f"Image file does not exist. Checked:\n{checked}")

    if path.suffix.lower() not in SUPPORTED_IMAGE_TYPES:
        raise ValueError(
            f"Unsupported image extension {path.suffix!r}. "
            f"Use one of: {sorted(SUPPORTED_IMAGE_TYPES)}"
        )
    return path


def normalize_model_name(model_name: str) -> str:
    if model_name in MODEL_PATHS:
        return model_name

    normalized = model_name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in MODEL_ALIASES:
        return MODEL_ALIASES[normalized]

    available = ", ".join(MODEL_PATHS)
    raise ValueError(f"Unknown model_name={model_name!r}. Choose one of: {available}")


def load_class_names() -> list[str]:
    global _class_names

    if _class_names is None:
        if not CLASS_NAMES_PATH.exists():
            raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")
        with open(CLASS_NAMES_PATH, encoding="utf-8") as file:
            _class_names = json.load(file)
    return _class_names


def load_tensorflow() -> Any:
    global _tf

    if _tf is None:
        import tensorflow as tensorflow

        _tf = tensorflow
    return _tf


def load_model(model_name: str) -> Any:
    canonical_name = normalize_model_name(model_name)
    if canonical_name not in _models:
        tf = load_tensorflow()
        model_path = MODEL_PATHS[canonical_name]
        if not model_path.exists():
            raise FileNotFoundError(f"{canonical_name} model not found: {model_path}")
        _models[canonical_name] = tf.keras.models.load_model(model_path)
    return _models[canonical_name]


def load_image_array(image_path: str | Path) -> tuple[np.ndarray, Path]:
    tf = load_tensorflow()
    path = resolve_image_path(image_path)
    image = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    array = tf.keras.utils.img_to_array(image).astype("float32")
    return array, path


def predict_batch(model: Any, images: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype="float32")
    return model.predict(images, verbose=0)


def build_prediction_result(
    model_name: str,
    image_path: str | Path,
    top_k: int,
) -> tuple[dict[str, Any], Any, np.ndarray]:
    canonical_name = normalize_model_name(model_name)
    class_names = load_class_names()
    model = load_model(canonical_name)
    image_array, resolved_path = load_image_array(image_path)
    image_batch = np.expand_dims(image_array, axis=0)

    probabilities = predict_batch(model, image_batch)[0]
    safe_top_k = max(1, min(int(top_k), len(class_names)))
    top_indices = np.argsort(probabilities)[-safe_top_k:][::-1]

    predictions = [
        {
            "rank": rank,
            "class": class_names[int(index)],
            "confidence": float(probabilities[int(index)]),
        }
        for rank, index in enumerate(top_indices, start=1)
    ]
    top_index = int(top_indices[0])

    result = {
        "model": canonical_name,
        "image": str(resolved_path),
        "top_class_index": top_index,
        "top_class": class_names[top_index],
        "top_confidence": float(probabilities[top_index]),
        "predictions": predictions,
    }
    return result, model, image_batch


def render_shap_png(
    model: Any,
    image_batch: np.ndarray,
    class_index: int,
    class_names: list[str],
    max_evals: int,
    batch_size: int,
) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    masker = shap.maskers.Image("inpaint_telea", image_batch[0].shape)
    explainer = shap.Explainer(
        lambda masked_images: predict_batch(model, masked_images),
        masker,
        output_names=class_names,
    )
    shap_values = explainer(
        image_batch,
        max_evals=max(50, int(max_evals)),
        batch_size=max(1, int(batch_size)),
        outputs=[int(class_index)],
    )

    plt.ioff()
    shap.image_plot(shap_values, show=False)
    figure = plt.gcf()
    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    plt.close(figure)
    return buffer.getvalue()


def load_vector_db() -> Any:
    global _vector_db

    if _vector_db is not None:
        return _vector_db

    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(
            f"Plant disease vector DB not found: {VECTOR_DB_DIR}. "
            "Build it with AAI_project/rag/build_vector_db.ipynb first."
        )

    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError(
            "Plant disease retrieval dependencies are missing. Install them with: "
            "pip install chromadb sentence-transformers langchain-chroma langchain-huggingface"
        ) from exc

    embedding = HuggingFaceEmbeddings(model_name=VECTOR_DB_EMBEDDING_MODEL)
    _vector_db = Chroma(
        collection_name=VECTOR_DB_COLLECTION_NAME,
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embedding,
    )
    return _vector_db


def build_retrieval_result(query: str, top_k: int) -> dict[str, Any]:
    vector_db = load_vector_db()
    safe_top_k = max(1, min(int(top_k), 10))
    matches = vector_db.similarity_search_with_score(query, k=safe_top_k)

    results = []
    for rank, (document, score) in enumerate(matches, start=1):
        metadata = document.metadata
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "content": document.page_content,
                "title": str(metadata.get("title", "")),
                "labels": str(metadata.get("labels", "")),
                "source_url": str(metadata.get("source_url", "")),
                "final_url": str(metadata.get("final_url", "")),
                "resource_id": str(metadata.get("resource_id", "")),
                "chunk_index": int(metadata.get("chunk_index", 0)),
            }
        )

    return {
        "query": query,
        "result_count": len(results),
        "results": results,
    }


@mcp.tool(output_schema=PREDICTION_OUTPUT_SCHEMA)
def predict_plant_disease(
    image_path: Annotated[str, Field(min_length=1, description="Local path to a leaf image file.")],
    model_name: Annotated[
        ModelName,
        Field(description="Already-trained model to use for prediction."),
    ] = "MobileNetV2",
    top_k: Annotated[int, Field(ge=1, le=10, description="Number of ranked predictions to return.")] = 3,
    include_shap: Annotated[
        bool,
        Field(description="Return a SHAP explanation PNG as MCP image content."),
    ] = False,
    shap_max_evals: Annotated[
        int,
        Field(ge=50, le=2000, description="SHAP perturbation budget; higher is slower."),
    ] = 300,
    shap_batch_size: Annotated[
        int,
        Field(ge=1, le=100, description="Batch size used while computing SHAP values."),
    ] = 20,
) -> ToolResult:
    """Classify a local plant leaf image and optionally return a SHAP explanation image."""
    result, model, image_batch = build_prediction_result(
        model_name=model_name,
        image_path=image_path,
        top_k=top_k,
    )

    if include_shap:
        class_names = load_class_names()
        shap_png = render_shap_png(
            model=model,
            image_batch=image_batch,
            class_index=result["top_class_index"],
            class_names=class_names,
            max_evals=shap_max_evals,
            batch_size=shap_batch_size,
        )
        result["shap"] = {
            "included": True,
            "mime_type": "image/png",
            "explained_class": result["top_class"],
            "max_evals": max(50, int(shap_max_evals)),
            "batch_size": max(1, int(shap_batch_size)),
        }
    else:
        result["shap"] = {"included": False}

    text = json.dumps(result, indent=2)
    content = [TextContent(type="text", text=text)]

    if include_shap:
        content.append(
            ImageContent(
                type="image",
                data=base64.b64encode(shap_png).decode("ascii"),
                mimeType="image/png",
            )
        )

    return ToolResult(content=content, structured_content=result)


@mcp.tool(output_schema=RETRIEVAL_OUTPUT_SCHEMA)
def retrieve_plant_disease_info(
    query: Annotated[
        str,
        Field(
            min_length=1,
            description="Plant, disease, symptom, prevention, care, or management question.",
        ),
    ],
    top_k: Annotated[
        int,
        Field(ge=1, le=10, description="Number of vector search results to return."),
    ] = 4,
) -> ToolResult:
    """Retrieve plant and plant disease information from the local Chroma vector DB."""
    result = build_retrieval_result(query=query, top_k=top_k)
    text = json.dumps(result, indent=2)
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=result,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
