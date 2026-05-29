"""Lazy resource loading for MCP prediction tools."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from plant_assistant import settings
from plant_assistant.mcp.constants import (
    IMG_SIZE,
    MODEL_ALIASES,
    MODEL_PATHS,
    PROJECT_DIR,
    SERVER_DIR,
    SUPPORTED_IMAGE_TYPES,
    WORKSPACE_DIR,
)


logger = logging.getLogger(__name__)

# Module-level caches prevent repeated imports/model loads during one server
# process while keeping startup lazy.
_tf: Any | None = None
_models: dict[str, Any] = {}
_class_names: list[str] | None = None


def resolve_image_path(image_path: str | Path) -> Path:
    """Resolve a user-provided image path against known project locations."""

    raw_path = Path(image_path).expanduser()
    candidates = [raw_path]

    if not raw_path.is_absolute():
        # Relative paths may come from the Streamlit app, a notebook, or a user
        # prompt, so check the working directory and project-specific anchors.
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
    """Convert supported aliases into the canonical model name."""

    if model_name in MODEL_PATHS:
        return model_name

    normalized = model_name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in MODEL_ALIASES:
        return MODEL_ALIASES[normalized]

    available = ", ".join(MODEL_PATHS)
    raise ValueError(f"Unknown model_name={model_name!r}. Choose one of: {available}")


def load_class_names() -> list[str]:
    """Load the disease class labels once from the configured JSON file."""

    global _class_names

    if _class_names is None:
        if not settings.CLASS_NAMES_PATH.exists():
            raise FileNotFoundError(f"Class names file not found: {settings.CLASS_NAMES_PATH}")
        logger.info("Loading class names from %s", settings.CLASS_NAMES_PATH)
        with open(settings.CLASS_NAMES_PATH, encoding="utf-8") as file:
            _class_names = json.load(file)
    return _class_names


def load_tensorflow() -> Any:
    """Import TensorFlow lazily and configure GPU memory growth when possible."""

    global _tf

    if _tf is None:
        logger.info("Importing TensorFlow")
        import tensorflow as tensorflow

        gpus = tensorflow.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tensorflow.config.experimental.set_memory_growth(gpu, True)
                logger.info("Enabled TensorFlow GPU memory growth for %s", gpu)
            except RuntimeError:
                logger.exception("Could not enable TensorFlow GPU memory growth for %s", gpu)
        _tf = tensorflow
    return _tf


def load_model(model_name: str) -> Any:
    """Load and cache one Keras model by canonical or alias name."""

    canonical_name = normalize_model_name(model_name)
    if canonical_name not in _models:
        tf = load_tensorflow()
        model_path = MODEL_PATHS[canonical_name]
        if not model_path.exists():
            raise FileNotFoundError(f"{canonical_name} model not found: {model_path}")
        logger.info("Loading %s model from %s", canonical_name, model_path)
        _models[canonical_name] = tf.keras.models.load_model(model_path)
    return _models[canonical_name]


def load_image_array(image_path: str | Path) -> tuple[np.ndarray, Path]:
    """Load an image file into the model's expected array shape."""

    tf = load_tensorflow()
    path = resolve_image_path(image_path)
    image = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    array = tf.keras.utils.img_to_array(image).astype("float32")
    return array, path
