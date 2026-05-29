"""Prediction, comparison, and SHAP helpers for MCP tools."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from plant_assistant.mcp.constants import MODEL_PATHS
from plant_assistant.mcp.resources import load_class_names, load_image_array, load_model, normalize_model_name


logger = logging.getLogger(__name__)


def predict_batch(model: Any, images: np.ndarray) -> np.ndarray:
    """Run model inference on a batch and return raw probabilities."""

    images = np.asarray(images, dtype="float32")
    return model.predict(images, verbose=0)


def build_ranked_prediction(
    *,
    model_name: str,
    probabilities: np.ndarray,
    class_names: list[str],
    top_k: int,
) -> dict[str, Any]:
    """Build a ranked top-k prediction dictionary from model probabilities."""

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
    return {
        "model": model_name,
        "top_class_index": top_index,
        "top_class": class_names[top_index],
        "top_confidence": float(probabilities[top_index]),
        "predictions": predictions,
    }


def build_prediction_result(
    model_name: str,
    image_path: str | Path,
    top_k: int,
) -> tuple[dict[str, Any], Any, np.ndarray]:
    """Run one model on one image and return the structured prediction result."""

    canonical_name = normalize_model_name(model_name)
    class_names = load_class_names()
    model = load_model(canonical_name)
    image_array, resolved_path = load_image_array(image_path)
    image_batch = np.expand_dims(image_array, axis=0)

    probabilities = predict_batch(model, image_batch)[0]
    result = build_ranked_prediction(
        model_name=canonical_name,
        probabilities=probabilities,
        class_names=class_names,
        top_k=top_k,
    )
    result["image"] = str(resolved_path)
    return result, model, image_batch


def build_model_comparison_result(image_path: str | Path, top_k: int) -> dict[str, Any]:
    """Run both local models on one image and summarize agreement."""

    class_names = load_class_names()
    image_array, resolved_path = load_image_array(image_path)
    image_batch = np.expand_dims(image_array, axis=0)
    safe_top_k = max(1, min(int(top_k), min(10, len(class_names))))

    model_results = []
    for model_name in MODEL_PATHS:
        # Reuse the same image batch for both models so the comparison differs
        # only by model weights, not preprocessing.
        model = load_model(model_name)
        probabilities = predict_batch(model, image_batch)[0]
        model_results.append(
            build_ranked_prediction(
                model_name=model_name,
                probabilities=probabilities,
                class_names=class_names,
                top_k=safe_top_k,
            )
        )

    agreement = model_results[0]["top_class"] == model_results[1]["top_class"]
    confidence_gap = abs(
        float(model_results[0]["top_confidence"]) - float(model_results[1]["top_confidence"])
    )
    result = {
        "image": str(resolved_path),
        "top_k": safe_top_k,
        "agreement": agreement,
        "models": model_results,
        "confidence_gap": confidence_gap,
        "summary": "models_agree" if agreement else "models_disagree",
    }
    logger.info(
        "Model comparison completed agreement=%s top_classes=%s confidence_gap=%.4f",
        agreement,
        ", ".join(str(model_result["top_class"]) for model_result in model_results),
        confidence_gap,
    )
    return result


def render_shap_png(
    model: Any,
    image_batch: np.ndarray,
    class_index: int,
    class_names: list[str],
    max_evals: int,
    batch_size: int,
) -> bytes:
    """Render a SHAP image explanation as PNG bytes for MCP image output."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    # SHAP masks image regions, asks the model for predictions, and plots the
    # regions that contributed to the selected class prediction.
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
