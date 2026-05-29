"""FastMCP tool registrations for plant disease assistant capabilities."""

from __future__ import annotations

import base64
import json
import logging
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from mcp.types import ImageContent, TextContent
from pydantic import Field

from plant_assistant import settings
from plant_assistant.mcp.constants import (
    COMPARISON_OUTPUT_SCHEMA,
    PREDICTION_OUTPUT_SCHEMA,
    RETRIEVAL_OUTPUT_SCHEMA,
    ModelName,
)
from plant_assistant.mcp.prediction import (
    build_model_comparison_result,
    build_prediction_result,
    render_shap_png,
)
from plant_assistant.mcp.resources import load_class_names
from plant_assistant.mcp.retrieval import build_retrieval_result


logger = logging.getLogger(__name__)

# FastMCP registers the tools and validates arguments against annotations and
# the schemas declared on each tool.
mcp = FastMCP(
    "PlantDiseaseAssistant",
    instructions=(
        "Use predict_plant_disease to classify local leaf images with the "
        "already-trained plant disease CNN models. Set include_shap=true only "
        "when the user asks for an explanation image or SHAP visualization. "
        "Use compare_plant_disease_models when the user asks for model agreement "
        "or uncertainty, or when the default MobileNetV2 prediction is below "
        "85% confidence. "
        "Use retrieve_plant_disease_info to look up symptoms, prevention, "
        "care, and management information from the local plant disease vector DB."
    ),
    strict_input_validation=True,
)


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
    ] = settings.DEFAULT_SHAP_MAX_EVALS,
    shap_batch_size: Annotated[
        int,
        Field(ge=1, le=100, description="Batch size used while computing SHAP values."),
    ] = settings.DEFAULT_SHAP_BATCH_SIZE,
) -> ToolResult:
    """Classify a local plant leaf image and optionally return a SHAP explanation image."""

    logger.info(
        "Tool predict_plant_disease start image=%s model=%s top_k=%s include_shap=%s",
        image_path,
        model_name,
        top_k,
        include_shap,
    )
    result, model, image_batch = build_prediction_result(
        model_name=model_name,
        image_path=image_path,
        top_k=top_k,
    )

    if include_shap:
        # SHAP is optional because it is much slower than the prediction itself
        # and produces an extra image payload.
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
        # MCP image content is base64 encoded; the UI later decodes and renders
        # it below the assistant response.
        content.append(
            ImageContent(
                type="image",
                data=base64.b64encode(shap_png).decode("ascii"),
                mimeType="image/png",
            )
        )

    logger.info(
        "Tool predict_plant_disease completed top_class=%s confidence=%.4f",
        result["top_class"],
        result["top_confidence"],
    )
    return ToolResult(content=content, structured_content=result)


@mcp.tool(output_schema=COMPARISON_OUTPUT_SCHEMA)
def compare_plant_disease_models(
    image_path: Annotated[str, Field(min_length=1, description="Local path to a leaf image file.")],
    top_k: Annotated[int, Field(ge=1, le=10, description="Number of ranked predictions per model.")] = 3,
) -> ToolResult:
    """Compare MobileNetV2 and Custom CNN predictions for a local plant leaf image."""

    logger.info("Tool compare_plant_disease_models start image=%s top_k=%s", image_path, top_k)
    result = build_model_comparison_result(image_path=image_path, top_k=top_k)
    text = json.dumps(result, indent=2)
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=result,
    )


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

    logger.info("Tool retrieve_plant_disease_info start top_k=%s query=%r", top_k, query)
    result = build_retrieval_result(query=query, top_k=top_k)
    text = json.dumps(result, indent=2)
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=result,
    )
